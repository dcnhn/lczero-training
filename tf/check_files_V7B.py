#!/usr/bin/env python3

import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from chunkparser import V7_VERSION, V7B_VERSION
from train import get_all_chunks


def collect_paths_from_cfg(cfg):
    """Collect dataset paths from a training YAML (like rescore_files)."""
    ds = cfg.get("dataset", {})
    paths = []

    if "input" in ds:
        paths.append(ds["input"])

    # Flatten and deduplicate
    flat = []
    for p in paths:
        if isinstance(p, list):
            flat.extend(p)
        elif p is not None:
            flat.append(p)

    seen = set()
    unique = []
    for p in flat:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def check_versions(filenames):
    """Worker: check that each file starts with V7/V7B version magic."""
    bad = []
    for fname in filenames:
        try:
            with gzip.open(fname, "rb") as f:
                header = f.read(4)
            if header not in (V7_VERSION, V7B_VERSION):
                bad.append(fname)
        except Exception:
            # Any error counts as non-V7
            bad.append(fname)
    return len(filenames), bad


def parallel_check_v7(chunks, n_workers=16, n_jobs=1000):
    if not chunks:
        print("No chunk files to check.")
        return

    print(f"Checking {len(chunks)} chunk files for V7/V7B versions...")

    # Split into jobs
    jobs = []
    for n in range(n_jobs):
        lo = n * len(chunks) // n_jobs
        hi = min((n + 1) * len(chunks) // n_jobs, len(chunks))
        if lo >= hi:
            continue
        jobs.append(chunks[lo:hi])

    total_files = 0
    bad_files = set()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(check_versions, job) for job in jobs]

        for i, fut in enumerate(as_completed(futures), 1):
            count, bad = fut.result()
            total_files += count
            bad_files.update(bad)
            print(f"Completed {i}/{len(futures)} jobs", end="\r", flush=True)

    print()
    print(f"Total files checked: {total_files}")
    if bad_files:
        print(f"Non-V7/V7B chunk files found: {len(bad_files)}")
        for fname in sorted(bad_files):
            print(fname)
    else:
        print("All chunk files are V7/V7B.")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel check that all chunks referenced in a YAML are V7/V7B."
    )
    parser.add_argument(
        "--cfg",
        type=argparse.FileType("r"),
        required=True,
        help="YAML configuration file (same as used for training)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes (default 16)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1000,
        help="Number of jobs to split work into (default 1000)",
    )

    args = parser.parse_args()
    cfg = yaml.safe_load(args.cfg.read())

    data_paths = collect_paths_from_cfg(cfg)
    if not data_paths:
        raise SystemExit(
            "No dataset input paths found in YAML (expected dataset.input)."
        )

    fast_chunk_loading = cfg.get("dataset", {}).get("fast_chunk_loading", False)

    chunks = []
    for p in data_paths:
        chunks.extend(get_all_chunks(p, fast=fast_chunk_loading))

    if not chunks:
        raise SystemExit(
            "No .gz chunk files found for paths: " + ", ".join(data_paths)
        )

    parallel_check_v7(chunks, n_workers=args.workers, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
