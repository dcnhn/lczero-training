import argparse
import os
import glob
import gzip
from chunkparser import \
    ChunkParser,        \
    struct_sizes,       \
    V6B_VERSION,        \
    V7B_VERSION,        \
    V7_VERSION,         \
    V6_VERSION,         \
    V5_VERSION,         \
    CLASSICAL_INPUT,    \
    V4_VERSION,         \
    V3_VERSION

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

def get_all_chunks(path):

    if isinstance(path, list):
        print("getting chunks for", path)
        chunks = []
        for i in path:
            chunks += get_all_chunks(i)
        return chunks
    else:
        chunks = []
        for d in glob.glob(path):
            chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count .gz training chunks.")
    parser.add_argument("paths", nargs="+", help="Paths or globs to chunk dirs")
    args = parser.parse_args()

    chunks = get_all_chunks(args.paths)

    corrupted_file = open("corrupted.txt", "a")

    for cnt, filename in enumerate(chunks):
        try:
            with gzip.open(filename, "rb") as chunk_file:
                chunk_file.seek(0)
                chunkdata = chunk_file.read()

        except Exception as e:
            print(f"GZIP error while reading {filename}: {e}")
            corrupted_file.write(filename + "\n")

        if cnt % 1000 == 0:
            print(f"Processed {cnt + 1} chunk of {len(chunks)}.")

    corrupted_file.close()
    print("chunks found:", len(chunks))
    print("Corrupted list saved to corrupted.txt")