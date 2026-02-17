#!/usr/bin/env python3

import re
import json
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import List
import requests

DEFAULT_BASE_URL = "https://storage.lczero.org/files/training_data/"

# Container class
@dataclass
class TarFileInfo:
    name: str
    url: str
    size_mb: float


def fetch_tar_files(base_url: str) -> List[TarFileInfo]:
    # Send HTTP GET request to base url
    resp = requests.get(base_url)

    # Raise error in case of negative server response
    resp.raise_for_status()

    # Get text from the response
    # Create empty list of entries
    html = resp.text
    entries = []

    # Iterate over all lines of the text
    for line in html.splitlines():
        # Check if the line contains the tar-extenion.
        # Skip if not
        if ".tar" not in line:
            continue

        # Extract filename by searching a pattern like:
        # href="data/archive.tar"
        # -----             ----
        # Quotes are excluded.
        match = re.search(r'href="([^"]+\.tar)"', line)
        if not match:
            continue
        name = match.group(1)

        nums = re.findall(r'(\d+)', line)
        if not nums:
            continue

        # Get the last number as size in bytes and convert to MB
        size_bytes = int(nums[-1])
        size_mb = size_bytes / (1024 * 1024)

        url = base_url + name
        entries.append(TarFileInfo(name, url, size_mb))

    return sorted(entries, key=lambda x: x.size_mb, reverse=True)

def extract_yyyymmdd(name: str) -> int | None:
    # matches ...--20240819-1917... or ...-20210924-...
    m = re.search(r'(\d{8})-', name)
    if m:
        return int(m.group(1))
    return None

def print_table(entries: List[TarFileInfo], max_entries: int) -> None:
    if not entries:
        print("No .tar files found.")
        return

    name_n = 8
    name_w = max(len(e.name) for e in entries)
    size_w = 8

    header = f"{'NO'.ljust(name_n)}  {'NAME'.ljust(name_w)}  {'SIZE_MB'.rjust(size_w)}  URL\n"
    print(header)

    for i, entry in enumerate(entries):
        print(f"{str(i+1).ljust(name_n)}  {entry.name.ljust(name_w)}  {entry.size_mb:>8.2f}  {entry.url}")
        if i >= max_entries - 1:
            break


def save_csv(entries: List[TarFileInfo], filename: str):
    if not entries:
        print(f"No entries to save for {filename}")
        return

    name_n = 8
    name_w = max(len(e.name) for e in entries)
    size_w = 8

    with open(filename, "w", newline="") as f:
        header = f"{'NO'.ljust(name_n)}  {'NAME'.ljust(name_w)}  {'SIZE_MB'.rjust(size_w)}  URL\n"
        f.write(header)

        for i, entry in enumerate(entries):
            line = (
                f"{str(i+1).ljust(name_n)}  "
                f"{entry.name.ljust(name_w)}  "
                f"{entry.size_mb:>8.2f}  "
                f"{entry.url}\n"
            )
            f.write(line)

    print(f"\nCSV saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List and analyze LCZero training .tar files"
    )
    parser.add_argument(
        "--lczero-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL to scrape (default: {DEFAULT_BASE_URL})",
    )

    parser.add_argument(
        "--save-top",
        type=int,
        default=10,
        help="Number of largest .tar URLs to save (default: 10)",
    )

    args = parser.parse_args()

    files = fetch_tar_files(args.lczero_url)

    # Sanity check: URL must start with DEFAULT_BASE_URL
    if not args.lczero_url.startswith(DEFAULT_BASE_URL):
        parser.error(f"--lczero-url must start with {DEFAULT_BASE_URL}")

    # Keep only files dated 2024-01-01 or newer
    cutoff = 20240101
    files = [f for f in files if (d := extract_yyyymmdd(f.name)) is not None and d >= cutoff]

    print_table(files, max_entries=100)

    save_csv(files, filename="lczero_all_tars.csv")

    with open(f"lczero_largest_{args.save_top}_tars.txt", "w") as f:
        for e in sorted(files, key=lambda x: x.size_mb, reverse=True)[:args.save_top]:
            f.write(e.url + "\n")
