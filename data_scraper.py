#!/usr/bin/env python3

import re
import json
import csv
from dataclasses import dataclass, asdict
from typing import List, Optional
import requests

BASE_URL = "https://storage.lczero.org/files/training_data/test75/"

# Container class
@dataclass
class TarFileInfo:
    name: str
    url: str
    size_mb: float


def fetch_tar_files(base_url: str = BASE_URL) -> List[TarFileInfo]:
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

        # Extract all numbers from the current line
        nums = re.findall(r'(\d+)', line)
        if not nums:
            continue
        
        # Get the last number as size in bytes and convert to MB
        size_bytes = int(nums[-1])
        size_mb = size_bytes / (1024 * 1024)

        # Create full URL and append to entries list
        url = base_url + name
        entries.append(TarFileInfo(name, url, size_mb))

    return sorted(entries, key=lambda x: x.name)


def print_table(entries: List[TarFileInfo], max_entries: int) -> None:
    if not entries:
        print("No .tar files found.")
        return

    name_w = max(len(e.name) for e in entries)
    size_w = 8

    header = f"{'NAME'.ljust(name_w)}  {'SIZE_MB'.rjust(size_w)}  URL\n"
    print(header)

    for i, entry in enumerate(entries):
        print(f"{entry.name.ljust(name_w)}  {entry.size_mb:>8.2f}  {entry.url}")
        if i >= max_entries - 1:
            break

def save_csv(entries: List[TarFileInfo], filename="lczero_test75_tars.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "url", "size_mb"])
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)
    print(f"\nCSV saved to {filename}")

if __name__ == "__main__":
    files = fetch_tar_files()
    large_files = [f for f in files if f.size_mb > 500]
    huge_files = [f for f in files if f.size_mb > 1000]
    print_table(large_files, max_entries=100)
    save_csv(sorted(files, key=lambda x: x.size_mb, reverse=True))
    save_csv(large_files, filename="lczero_test75_tars_over_500MB.csv")
    save_csv(huge_files, filename="lczero_test75_tars_over_1000MB.csv")

    # Create a text file with the largest 50 files
    with open("largest_50_files.txt", "w") as f:
        for e in sorted(files, key=lambda x: x.size_mb, reverse=True)[:50]:
            f.write(e.url + "\n")