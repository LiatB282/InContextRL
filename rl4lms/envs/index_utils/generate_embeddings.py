#!/usr/bin/env python3

import glob
import os
import ntpath

import argparse
import logging
import pickle
import time
from typing import List, Tuple
import numpy as np
import json
import re
import sys
from sentence_transformers import SentenceTransformer
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import print_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

embeddings_file_format = r"encoded_passages_(.*).json"
dataset_file_format  = r"dataset_passages_(.*).pkl"
    
def load_base_model_embeddings(base_model_embeds_dir):
        id_to_embed = {}
        input_files = glob.glob(base_model_embeds_dir)

        for i, input_file in enumerate(input_files):
            file_name = ntpath.basename(input_file)

            if int(re.match(embeddings_file_format, file_name).group(1)) % args.workers != args.worker_number:
                continue

            logger.info(f"Reading file {i+1}/{len(input_files)}: {input_file}")
            with open(input_file, 'rb') as f:
                embeds =  pickle.load(f)

            for e in embeds:
                id_to_embed[e[0]] = e[1]

        return id_to_embed

def gen_ctx_vectors(
    ctx_rows,
    batch_size,
    model
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    total = 0
    results = [None] * n

    logger.info("Sorting contexts")
    sorted_indices = list(reversed(np.argsort([len(ctx) for _, ctx, _ in ctx_rows])))

    logger.info("Generating embeddings...")
    start_time = time.time()
    for j, batch_start in enumerate(range(0, n, batch_size)):
        batch_indices = sorted_indices[batch_start : batch_start + batch_size]
        batch_rows = [ctx_rows[ind] for ind in batch_indices]
        batch_ids, batch_inputs = zip(*batch_rows)
        batch_ids = list(batch_ids)

        out = torch.nn.functional.normalize(model.encode(batch_inputs), dim=1) 
        out = out.cpu()

        assert len(batch_ids) == out.size(0)

        total += len(batch_ids)

        for i, ind in enumerate(batch_indices):
            assert results[ind] is None
            results[ind] = ((batch_ids[i], batch_inputs[i]), out[i].view(-1).numpy())

        if total % 10 == 0:
            logger.info(f"Encoded {total} passages, took {time.time()-start_time:.1f} seconds")

    logger.info(f"Done. Took {(time.time()-start_time)/60:.1f} minutes")

    return results

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print_args(args, args.output_dir)

    model = SentenceTransformer(args.model_name, cache_dir=args.cache_dir).to(args.device)
    model.eval()

    input_files = glob.glob(args.input_files)
    logger.info(f"Processing {len(input_files)} files.")
    total_num_psgs = 0

    for i, input_file in enumerate(input_files):
        file_name = ntpath.basename(input_file)
        if int(re.match(dataset_file_format, file_name).group(1)) % args.workers != args.worker_number:
            continue

        out_file = os.path.join(args.output_dir, file_name.replace("dataset", "encoded").replace("json", "pkl"))
        if os.path.exists(out_file):
            continue

        logger.info(f"Processing file: {input_file}")

        with open(input_file, "rb") as f:
            rows = json.load(f)

        data = gen_ctx_vectors(rows, args.batch_size, model)

        assert not os.path.exists(out_file)
        logger.info("Writing results to %s" % out_file)
        with open(out_file, mode="wb") as f:
            pickle.dump(data, f)
        total_num_psgs += len(data)
        logger.info(f"Total passages processed {total_num_psgs} from {i+1} files. Written to {out_file}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to dataset with rows of [id, ctx]"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the passage encoder forward pass",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default='t5-base'
    )

    args = parser.parse_args()

    args.input_files = f"{args.input_dir}/tokenized_passages_*.pkl"

    if "workers" in os.environ and "worker_number" in os.environ:
        args.workers = int(os.environ["workers"])
        args.worker_number = int(os.environ["worker_number"])
    else:
        args.workers = 1
        args.worker_number = 0

    main(args)