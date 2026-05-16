#!/usr/bin/env python3
# extract kernel8.img and dtb from a fat32 boot partition image, with lfn support
import struct, os

IMG = "boot.img"
OUTDIR = "boot_files"
os.makedirs(OUTDIR, exist_ok=True)

# files we want to extract (lowercase for matching)
TARGETS = {"kernel8.img", "bcm2710-rpi-3-b-plus.dtb", "bcm2710-rpi-3-b.dtb",
           "cmdline.txt", "config.txt"}

with open(IMG, "rb") as f:
    data = f.read()

# parse bpb
bytes_per_sector = struct.unpack_from("<H", data, 11)[0]
sectors_per_cluster = data[13]
reserved_sectors = struct.unpack_from("<H", data, 14)[0]
num_fats = data[16]
sectors_per_fat = struct.unpack_from("<I", data, 36)[0]
root_cluster = struct.unpack_from("<I", data, 44)[0]
cluster_size = bytes_per_sector * sectors_per_cluster
fat_offset = reserved_sectors * bytes_per_sector
data_offset = fat_offset + num_fats * sectors_per_fat * bytes_per_sector

def cluster_to_offset(cluster):
    return data_offset + (cluster - 2) * cluster_size

def get_fat_chain(start_cluster):
    chain = []
    c = start_cluster
    while c < 0x0FFFFFF8 and c >= 2:
        chain.append(c)
        c = struct.unpack_from("<I", data, fat_offset + c * 4)[0] & 0x0FFFFFFF
        if len(chain) > 200000:
            break
    return chain

def decode_lfn_entry(ent):
    # extract ucs-2 characters from an lfn directory entry
    chars = []
    for off in [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30]:
        ch = struct.unpack_from("<H", ent, off)[0]
        if ch == 0 or ch == 0xFFFF:
            break
        chars.append(chr(ch))
    return "".join(chars)

def read_dir(cluster):
    # read directory entries with lfn support
    entries = []
    chain = get_fat_chain(cluster)
    lfn_parts = {}

    for c in chain:
        off = cluster_to_offset(c)
        for i in range(cluster_size // 32):
            ent = data[off + i*32 : off + (i+1)*32]
            if ent[0] == 0:
                return entries
            if ent[0] == 0xE5:
                lfn_parts = {}
                continue
            attr = ent[11]
            # lfn entry: attr == 0x0f
            if attr == 0x0F:
                seq = ent[0] & 0x3F
                lfn_parts[seq] = decode_lfn_entry(ent)
                continue

            # regular entry - reconstruct long name
            short_name = ent[0:8].decode("ascii", errors="replace").rstrip()
            short_ext = ent[8:11].decode("ascii", errors="replace").rstrip()
            short_fname = f"{short_name}.{short_ext}" if short_ext else short_name

            if lfn_parts:
                long_name = ""
                for k in sorted(lfn_parts.keys()):
                    long_name += lfn_parts[k]
                lfn_parts = {}
            else:
                long_name = short_fname

            hi = struct.unpack_from("<H", ent, 20)[0]
            lo = struct.unpack_from("<H", ent, 26)[0]
            start = (hi << 16) | lo
            size = struct.unpack_from("<I", ent, 28)[0]
            entries.append((long_name, attr, start, size))
    return entries

def extract_file(start_cluster, size, outpath):
    chain = get_fat_chain(start_cluster)
    with open(outpath, "wb") as f:
        remaining = size
        for c in chain:
            off = cluster_to_offset(c)
            chunk = min(cluster_size, remaining)
            f.write(data[off:off+chunk])
            remaining -= chunk
            if remaining <= 0:
                break
    print(f"  -> Extracted: {outpath} ({size:,} bytes)")

print("Root directory:")
entries = read_dir(root_cluster)
for fname, attr, start, size in entries:
    is_dir = "DIR" if attr & 0x10 else "   "
    print(f"  {is_dir} {fname:40s} cluster={start:6d}  size={size:>10,}")
    if fname.lower() in TARGETS:
        extract_file(start, size, os.path.join(OUTDIR, fname.lower()))

print("\nDone.")
