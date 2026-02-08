#!/usr/bin/env python3

import argparse
import os

import yaml

from chunkparser import rescore
from train import get_all_chunks


def collect_paths_from_cfg(cfg):
    ds = cfg.get("dataset", {})
    paths = []

    # Primary inputs (most configs use this)
    if "input" in ds:
        paths.append(ds["input"])

    # Flatten and deduplicate
    flat = []
    for p in paths:
        if isinstance(p, list):
            flat.extend(p)
        elif p is not None:
            flat.append(p)
    # Preserve order but remove duplicates
    seen = set()
    unique = []
    for p in flat:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def main(args):
    cfg = yaml.safe_load(args.cfg.read())

    data_paths = collect_paths_from_cfg(cfg)
    if not data_paths:
        raise SystemExit("No dataset input paths found in YAML (dataset.input / input_train / input_test / input_validation).")

    fast_chunk_loading = cfg.get("dataset", {}).get("fast_chunk_loading", False)

    # Collect all chunk filenames using the same helper as train.py
    all_chunks = []
    for p in data_paths:
        chunks = get_all_chunks(p, fast=fast_chunk_loading)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No .gz chunk files found for paths: " + ", ".join(data_paths))

    print(f"Found {len(all_chunks)} chunk files to rescore.")

    # Optional overrides for workers/jobs
    n_workers = args.workers if args.workers is not None else 16
    n_jobs = args.jobs if args.jobs is not None else 1000

    # Forward any extra kwargs (e.g. custom alphas) if desired later
    rescore(all_chunks, n_workers=n_workers, n_jobs=n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescore LCZero chunk files based on YAML config paths.")
    parser.add_argument("--cfg", type=argparse.FileType("r"), required=True,
                        help="YAML configuration file (same as used for training).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes for rescore (default 16).")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of jobs to split work into (default 1000).")

    main(parser.parse_args())
