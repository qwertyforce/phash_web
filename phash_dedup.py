import lmdb
from os.path import exists
from os import remove
import faiss
import numpy as np
import imagesize
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str,nargs='?', default="./../test_images/")
args = parser.parse_args()
distance_threshold = 64

IMAGE_PATH = args.image_path

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(4, 'big')

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

if exists("./data/populated.index"):
    index = faiss.read_index_binary("./data/populated.index")
else:
    print("Index is not found! Exiting...")
    exit()

if exists("./data/id_to_filename.lmdb") and exists("./data/filename_to_id.lmdb"):
    DB_id_to_filename = lmdb.open('./data/data/id_to_filename.lmdb',map_size=50*1_000_000) #50mb
    DB_filename_to_id = lmdb.open('./data/data/filename_to_id.lmdb',map_size=50*1_000_000) #50mb
else:
    print("DB_id_to_filename is not found! Exiting...")
    exit()

DB_phash = lmdb.open('./data/phashes.lmdb',map_size=500*1_000_000) #500mb

def get_file_name_by_id(image_id):
    with DB_id_to_filename.begin() as txn:
        file_name = txn.get(int_to_bytes(image_id), default=False)
        if not file_name:
            return False
        return file_name.decode("utf-8")

to_remove_id_set = set()
to_remove_filename_set = set()
all_ids = []
with DB_phash.begin(buffers=True) as txn:
    with txn.cursor() as curs:
        for key in curs.iternext(keys=True, values=False):
            all_ids.append(int_from_bytes(key))


all_ids = [all_ids[i:i + 64] for i in range(0, len(all_ids), 64)]
arr_results = []

for batch in tqdm(all_ids):
    target_features = np.array([index.reconstruct(id) for id in batch if id not in to_remove_id_set])
    if target_features.shape[0] == 0:
        continue
    lims, D, I = index.range_search(target_features, distance_threshold)
    I = [int(el) for el in I]
    D = [int(el) for el in D]
    for i in range(len(lims)-1):
        if batch[i] in to_remove_id_set:
            continue
        image_ids = I[lims[i]:lims[i+1]]
        image_ids = [el for el in image_ids if el not in to_remove_id_set]
        if len(image_ids) <=  1:
            continue
        # print("==============")
        # print(f"similar images: {image_ids}")
        # print(f"distances: {D[lims[i]:lims[i+1]]}")
        file_names = []
        resolutions = []
        max_pixels = -1
        id_img_to_keep = -1

        file_names = [get_file_name_by_id(image_id) for image_id in image_ids]
        for image_id, file_name in zip(image_ids, file_names):
            if not file_name:
                print("not file_name. Something went wrong!!!")
                continue
            img_path = IMAGE_PATH+file_name
            width, height = imagesize.get(img_path)
            resolutions.append([width,height])
            if width*height>max_pixels:
                id_img_to_keep=image_id
                max_pixels = width*height

        for image_id, file_name in zip(image_ids, file_names):
            if image_id == id_img_to_keep:
                continue
            to_remove_id_set.add(image_id)
            to_remove_filename_set.add(file_name)
        arr_results.append([file_names,D[lims[i]:lims[i+1]],resolutions])

print(len(to_remove_id_set))

import json
with open("phash.res","w") as file:
    json.dump(arr_results, file)

print("removing duplicates from index")
index.remove_ids(np.int64(list(to_remove_id_set)))

print("removing duplicates from DB_id_to_filename and DB_phash")
to_remove_id_set = [int_to_bytes(image_id) for image_id in to_remove_id_set]

with DB_phash.begin(write=True,buffers=True) as txn:
    for image_id in tqdm(to_remove_id_set):
        txn.delete(image_id)
with DB_id_to_filename.begin(write=True,buffers=True) as txn:
    for image_id in tqdm(to_remove_id_set):
        txn.delete(image_id)

print("removing duplicates from DB_filename_to_id and deleting images")   
with DB_filename_to_id.begin(write=True,buffers=True) as txn:
    for file_name in tqdm(to_remove_filename_set):
        remove(IMAGE_PATH+file_name)
        txn.delete(file_name.encode())  

print("updating populated.index...")
faiss.write_index_binary(index, "./data/populated.index")