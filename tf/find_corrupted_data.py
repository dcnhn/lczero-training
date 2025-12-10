import argparse
import glob
import gzip
import struct
from tqdm import tqdm
from chunkparser import struct_sizes, V6_VERSION

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

def in_range(root_q, st_q, root_d):
    e = 1e-2
    return (-1.0 - e <= root_q <= 1.0 + e) and (-1.0 - e <= st_q <= 1.0 + e) and (0.0 - e <= root_d <= 1.0 + e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count .gz training chunks.")
    parser.add_argument("paths", nargs="+", help="Paths or globs to chunk dirs")
    args = parser.parse_args()

    chunks = get_all_chunks(args.paths)

    corrupted_file = open("corrupted.txt", "a")

    num_positions = 0
    for cnt, filename in tqdm(enumerate(chunks), total=len(chunks), desc="Scanning chunks"):
        try:
            with gzip.open(filename, "rb") as chunk_file:
                chunk_file.seek(0)
                chunkdata = chunk_file.read()
                version = chunkdata[0:4]
                record_size = struct_sizes.get(version, None)
                if record_size is None:
                    print("Unknown chunk version in file:", filename)
                    corrupted_file.write(filename + "\n")
                    continue

                if not (version == V6_VERSION):
                    print(F"File {filename} is not V6 version.")
                    corrupted_file.write(filename + "\n")
                    continue

                skip_file = False
                for record_start_bit in range(0, len(chunkdata), record_size):
                    # Get individual record
                    record = chunkdata[record_start_bit:record_start_bit + record_size]

                    # Get root_q and root_d (V6 layout) and validate
                    root_q = struct.unpack("f", record[8280:8284])[0]
                    root_d = struct.unpack("f", record[8288:8292])[0]
                    st_q = struct.unpack("f", record[8352:8356])[0]

                    # Check validity of root_q, st_q and root_d
                    if not in_range(root_q, st_q, root_d):
                        print(F"Corrupted data in file {filename}: root_q={root_q}, st_q={st_q}, root_d={root_d}")
                        corrupted_file.write(filename + "\n")
                        skip_file = True

                        # Break inner loop to avoid multiple entries for the same file
                        break

                if skip_file:
                    continue

                # Only increment number of positions if file is valid
                n_chunks = len(chunkdata) // record_size
                num_positions += n_chunks

        except Exception as e:
            print(f"GZIP error while reading {filename}: {e}")
            corrupted_file.write(filename + "\n")

    corrupted_file.close()
    print("chunks found:", len(chunks), "total positions:", num_positions)
    print("Corrupted list saved to corrupted.txt")