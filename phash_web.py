import uvicorn
if __name__ == '__main__':
    uvicorn.run('phash_web:app', host='127.0.0.1', port=33336, log_level="info")

import asyncio
import faiss
from typing import Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, Response, status, HTTPException
from os.path import exists
import numpy as np
from scipy.fft import dct
from numba import jit
import cv2
import lmdb
DB = lmdb.open('./phashes.lmdb',map_size=500*1_000_000) #500mb
index = None

DATA_CHANGED_SINCE_LAST_SAVE = False
# from concurrent.futures import ThreadPoolExecutor
# SearchThreadExecutor = ThreadPoolExecutor(max_workers=4)
# from joblib import Parallel, delayed

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes((x.bit_length() + 7) // 8, 'big')

def init_index():
    global index
    if exists("./populated.index"):
        index = faiss.read_index_binary("./populated.index")
    else:
        print("Index is not found! Exiting...")
        exit()


def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)


@jit(cache=True, nopython=True)
def bit_list_to_72_uint8(bit_list_576):
    uint8_arr = []
    for i in range(len(bit_list_576)//8):
        bit_list = []
        for j in range(8):
            if(bit_list_576[i*8+j] == True):
                bit_list.append(1)
            else:
                bit_list.append(0)
        uint8_arr.append(bit_list_to_int(bit_list))
    return np.array(uint8_arr, dtype=np.uint8)


@jit(cache=True, nopython=True)
def bit_list_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


@jit(cache=True, nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()


def fast_phash(resized_image, hash_size):
    dct_data = dct(dct(resized_image, axis=0), axis=1)
    return diff(dct_data, hash_size)


def get_phash(image_buffer, hash_size=24, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    query_image = cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_GRAYSCALE)
    query_image = cv2.resize(query_image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    bit_list_576 = fast_phash(query_image, hash_size)
    phash = bit_list_to_72_uint8(bit_list_576)
    return phash


def get_phash_and_mirrored_phash(image_buffer, hash_size=24, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    query_image = cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_GRAYSCALE)
    query_image = cv2.resize(query_image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    mirrored_query_image = cv2.flip(query_image, 1)
    bit_list_576 = fast_phash(query_image, hash_size)
    bit_list_576_mirrored = fast_phash(mirrored_query_image, hash_size)
    phash = bit_list_to_72_uint8(bit_list_576)
    mirrored_phash = bit_list_to_72_uint8(bit_list_576_mirrored)
    return np.array([phash, mirrored_phash])


def delete_descriptor_by_id(id):
    with DB.begin(write=True,buffers=True) as txn:
        txn.delete(int_to_bytes(id))   #True = deleted False = not found


def add_descriptor(id, phash):
    with DB.begin(write=True, buffers=True) as txn:
        txn.put(int_to_bytes(id),np.frombuffer(phash,dtype=np.uint8))


def phash_reverse_search(target_features,k,distance_threshold):
    if k is not None:
        D, I = index.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        _, D, I = index.range_search(target_features, distance_threshold)

    _, indexes = np.unique(I, return_index=True)
    res=[{"image_id":int(I[idx]), "distance":int(D[idx])} for idx in indexes]
    res = sorted(res, key=lambda x: x["distance"])
    if k:
        return res[:k]
    return res


app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_image_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    distance_threshold: Union[str,int,None] = None

@app.post("/phash_get_similar_images_by_id")
async def phash_get_similar_images_by_id_handler(item: Item_image_id):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        if item.k:
            k = int(k)
        if item.distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = index.reconstruct(item.image_id).reshape(1,-1)
        # loop = asyncio.get_event_loop()
        # similar = await loop.run_in_executor(SearchThreadExecutor,
        #                                     phash_reverse_search, target_features,
        #                                     k, distance_threshold)
        similar = phash_reverse_search(target_features,k,distance_threshold)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Error in phash_get_similar_images_by_id_handler")

@app.post("/phash_get_similar_images_by_image_buffer")
async def phash_get_similar_images_by_image_buffer_handler(image: bytes = File(...), k: Optional[str] = Form(None), distance_threshold: Optional[str] = Form(None)):
    try:
        if k:
            k=int(k)
        if distance_threshold:
            distance_threshold=int(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")
        target_features = get_phash_and_mirrored_phash(image) #TTA
        # loop = asyncio.get_event_loop()
        # similar = await loop.run_in_executor(SearchThreadExecutor,
        #                                      phash_reverse_search, target_features,
        #                                      k, distance_threshold)
        similar = phash_reverse_search(target_features,k,distance_threshold)
        # print(similar)
        return similar
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Error in phash_get_similar_images_by_image_buffer")


@app.post("/calculate_phash_features")
async def calculate_phash_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        features = get_phash(image)
        add_descriptor(int(image_id),features)
        index.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't calculate phash features")

@app.post("/delete_phash_features")
async def delete_phash_features_handler(item: Item_image_id):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        delete_descriptor_by_id(item.image_id)
        res = index.remove_ids(np.int64([item.image_id]))
        if res != 0: 
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {item.image_id}")    
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't delete phash features")
print(__name__)

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index_binary(index, "./populated.index")
    loop.call_later(10, periodically_save_index,loop)

if __name__ == 'phash_web':
    init_index()
    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)
