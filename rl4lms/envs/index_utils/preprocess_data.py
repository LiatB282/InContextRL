import argparse
import csv
import logging
import os
import pickle
import time
from multiprocessing.pool import Pool
import json
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def process_single_shard(rows, output_file):
    results = []
    for example in rows:
        results.append((example["id"], example["input"]))
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


def prepare_params_for_shard(output_dir, rows, num_shards, shard_idx):
    assert 0 <= shard_idx < num_shards
    shard_size = int(len(rows) / num_shards)
    shard_start = shard_idx * shard_size
    if shard_idx == (num_shards - 1):
        shard_rows = rows[shard_start:]
    else:
        shard_rows = rows[shard_start:shard_start+shard_size]

    output_file = os.path.join(output_dir, f"dataset_passages_{shard_idx}.pkl")
    return shard_rows, output_file

def read_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % path)
        data = json.load(f)
    logger.info("Data size: {}".format(len(data)))
    return data

def main(args):    
    logger.info("Reading data")
    start_time = time.time()

    for split in ["train", "validation", "test"]:
        logger.info(f"Working on split {split}")
        output_dir = f"{args.output_dir}/{split}"
        samples = load_dataset('trivia_qa', 'rc', split)

        rows = [ {"id": i+1, "input": f"Input: {samples[i]['question']} Output: {samples[i]['answer']['value']}"} for i in range(len(samples))]

        logger.info(f"Done. Took {(time.time() - start_time) / 60:.1f} minutes")

        # Creating the output directory. If exists, crash.
        os.makedirs(args.output_dir, exist_ok=False)

        params = [prepare_params_for_shard(output_dir, rows,
                                        args.num_processes_and_shards,
                                        shard_idx) for shard_idx in range(args.num_processes_and_shards)]
        with Pool(args.num_processes_and_shards if args.num_processes_and_shards else None) as p:
            p.starmap(process_single_shard, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_processes_and_shards", type=int, default=10)
    parser.add_argument("--output_dir", required=True, type=str)

    args = parser.parse_args()

    main(args)