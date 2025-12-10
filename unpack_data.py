import os
import tarfile
import urllib.request

urls_file = "datasets.txt"

with open(urls_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    filename = url.split("/")[-1]
    folder = filename.replace(".tar", "")
    print(f"Processing URL: {url} with filename {filename}")

    # Skip if already extracted
    if os.path.isdir(folder):
        print(f"Skipping {folder} (folder exists)")
        continue

    # Skip if tar file not downloaded
    if not os.path.isfile(filename):
        print(f"Skipping {filename} (file not found)")
        continue

    # Extract tar
    print(f"Extracting {filename} ...")
    with tarfile.open(filename, "r") as tar:
        tar.extractall()

print("Done.")
