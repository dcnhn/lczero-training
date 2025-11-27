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
    num_positions = 0
    for cnt, filename in enumerate(chunks):
        with gzip.open(filename, "rb") as chunk_file:
            if cnt % 200 == 0:
                print("reading chunk", cnt + 1, "of", len(chunks), ":", filename)
            chunk_file.seek(0)
            chunkdata = chunk_file.read()
            version = chunkdata[0:4]
            record_size = struct_sizes.get(version, None)
            n_chunks = len(chunkdata) // record_size
            num_positions += n_chunks

    print("chunks found:", len(chunks), "total positions:", num_positions)