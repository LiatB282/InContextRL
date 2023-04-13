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
from transformers import AutoTokenizer, T5EncoderModel
from InstructorEmbedding import INSTRUCTOR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import print_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

embeddings_file_format = r"encoded_passages_(.*).pkl"
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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def gen_ctx_vectors(
    ctx_rows,
    batch_size,
    model,
    tokenizer,
    is_instructor,
    is_last=False
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    total = 0
    if is_last:
        results = [None] * (n + 1)
    else:
        results = [None] * n

    logger.info("Sorting contexts")
    sorted_indices = list(reversed(np.argsort([len(ctx) for _, ctx in ctx_rows])))

    logger.info("Generating embeddings...")
    start_time = time.time()
    for j, batch_start in enumerate(range(0, n, batch_size)):
        batch_indices = sorted_indices[batch_start : batch_start + batch_size]
        batch_rows = [ctx_rows[ind] for ind in batch_indices]
        batch_ids, batch_inputs = zip(*batch_rows)
        batch_ids = list(batch_ids)

        model_input = tokenizer.batch_encode_plus(list(batch_inputs), return_tensors="pt", padding='longest', max_length=512).to('cuda')#['input_ids'].squeeze(0)


        if is_instructor:
            instruction = "Represent the question answering example:"
            encodings = model.encode([[instruction,sentence] for sentence in batch_inputs])
        else:
            with torch.no_grad():
                outputs = model(**model_input)
            encodings = mean_pooling(outputs, model_input['attention_mask'])

        tokenized = model_input['input_ids'].squeeze(0)
        #encodings = torch.from_numpy(encodings)

        out = torch.nn.functional.normalize(encodings, dim=1)
        out = out.cpu()

        assert len(batch_ids) == out.size(0)

        total += len(batch_ids)

        for i, ind in enumerate(batch_indices):
            assert results[ind] is None
            results[ind] = ((batch_ids[i], tokenized[i]), out[i].view(-1).numpy())

        if total % 10 == 0:
            logger.info(f"Encoded {total} passages, took {time.time()-start_time:.1f} seconds")

    # Adding the ending action
    if is_last:
        logger.info("Adding the ending action")
        results[n] = ((0, np.zeros(1)), np.zeros(768))

    logger.info(f"Done. Took {(time.time()-start_time)/60:.1f} minutes")

    return results

def main(args):
    print_args(args)

    splits = ["train", "validation", "test"]

    for split in splits:
        logger.info(f"Working on split {split}")
        output_dir = f"{args.output_dir}/{split}"
        input_files_path = f"{args.input_dir}/{split}/dataset_passages_*.pkl"

        os.makedirs(output_dir, exist_ok=True)

        if args.use_instructor:
            model = INSTRUCTOR('hkunlp/instructor-base')
        else:
            #model = SentenceTransformer(args.model_name, cache_folder=args.cache_dir).to(args.device)
            model = T5EncoderModel.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(args.device)
        
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/home/gamir/liat/cache")

        input_files = glob.glob(input_files_path)
        logger.info(f"Processing {len(input_files)} files.")
        total_num_psgs = 0

        for i, input_file in enumerate(input_files):
            file_name = ntpath.basename(input_file)
            if int(re.match(dataset_file_format, file_name).group(1)) % args.workers != args.worker_number:
                continue

            out_file = os.path.join(output_dir, file_name.replace("dataset", "encoded"))
            if os.path.exists(out_file):
                continue

            logger.info(f"Processing file: {input_file}")

            with open(input_file, "rb") as f:
                rows = pickle.load(f)

            is_last = i == len(input_files) - 1
            data = gen_ctx_vectors(rows, args.batch_size, model, tokenizer, args.use_instructor, is_last=is_last)

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
        default='sentence-transformers/gtr-t5-base'
    )

    parser.add_argument(
        "--use_instructor",
        action="store_true",
        help="Use lexical enrichment"
    )

    args = parser.parse_args()

    if "workers" in os.environ and "worker_number" in os.environ:
        args.workers = int(os.environ["workers"])
        args.worker_number = int(os.environ["worker_number"])
    else:
        args.workers = 1
        args.worker_number = 0

    main(args)