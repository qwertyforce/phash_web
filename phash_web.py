import uvicorn
if __name__ == '__main__':
    uvicorn.run('phash_web:app', host='127.0.0.1', port=33336, log_level="info")
    exit()

from os import environ
if not "GET_FILENAMES" in environ:
    print("GET_FILENAMES not found! Defaulting to 0...")
    GET_FILENAMES = 0
else:
    if environ["GET_FILENAMES"] not in ["0","1"]:
        print("GET_FILENAMES has wrong argument! Defaulting to 0...")
        GET_FILENAMES = 0
    else:
        GET_FILENAMES = int(environ["GET_FILENAMES"])

import traceback
import asyncio
import faiss
from typing import Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, Response, status, HTTPException
from os.path import exists
import numpy as np
from PIL import Image
import io

from modules.lmdb_ops import get_dbs
from modules.byte_ops import int_to_bytes, int_from_bytes
from modules.phash_ops import get_phash_by_image

index = None    
DATA_CHANGED_SINCE_LAST_SAVE = False
app = FastAPI()

def main():
    global DB_phash, DB_filename_to_id, DB_id_to_filename
    init_index()
    DB_phash, DB_filename_to_id, DB_id_to_filename = get_dbs()

    loop = asyncio.get_event_loop()
    loop.call_later(10, periodically_save_index,loop)



def read_img_buffer(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    if img.mode != 'L':
        img = img.convert('L')
    return img

def get_phash(image_buffer, hash_size=24, highfreq_factor=4):
    query_image = read_img_buffer(image_buffer)
    phash = get_phash_by_image(query_image,hash_size, highfreq_factor)
    return phash

def get_phash_and_mirrored_phash(image_buffer, hash_size=24, highfreq_factor=4):
    query_image = read_img_buffer(image_buffer)
    phashes = get_phash_by_image(query_image,hash_size, highfreq_factor, aug=True)
    return phashes

def check_if_exists_by_image_id(image_id):
    with DB_phash.begin(buffers=True) as txn:
        x = txn.get(int_to_bytes(image_id), default=False)
        if x:
            return True
        return False

def get_filenames_bulk(image_ids):
    image_ids_bytes = [int_to_bytes(x) for x in image_ids]

    with DB_id_to_filename.begin(buffers=False) as txn:
        with txn.cursor() as curs:
            file_names = curs.getmulti(image_ids_bytes)
    for i in range(len(file_names)):
        file_names[i] = file_names[i][1].decode()

    return file_names

def get_image_id_by_filename(file_name):
    with DB_filename_to_id.begin(buffers=True) as txn:
        image_id = txn.get(file_name.encode(), default=False)
        if not image_id:
            return False
        return int_from_bytes(image_id)
            
def delete_descriptor_by_id(image_id):
    image_id_bytes = int_to_bytes(image_id)
    with DB_phash.begin(write=True, buffers=True) as txn:
        txn.delete(image_id_bytes)   #True = deleted False = not found

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        file_name_bytes = txn.get(image_id_bytes, default=False)
        txn.delete(image_id_bytes)  

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.delete(file_name_bytes) 

def add_descriptor(image_id, phash):
    file_name_bytes = f"{image_id}.online".encode()
    image_id_bytes = int_to_bytes(image_id)
    with DB_phash.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, phash.tobytes())

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
        txn.put(image_id_bytes, file_name_bytes)

    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        txn.put(file_name_bytes, image_id_bytes)
    

def phash_reverse_search(target_features,k,distance_threshold):
    if k is not None:
        D, I = index.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        _, D, I = index.range_search(target_features, distance_threshold)

    idx_sorted_by_distance = np.argsort(D)
    res = []
    used_ids = set()
    for idx in idx_sorted_by_distance:
        if I[idx] not in used_ids:
            res.append({"image_id":int(I[idx]), "distance":int(D[idx])})
            used_ids.add(I[idx])

    if k:
        return res[:k]
    return res


@app.get("/")
async def read_root():
    return {"Hello": "World"}

class Item_phash_get_similar_images_by_id(BaseModel):
    image_id: int
    k: Union[str,int,None] = None
    distance_threshold: Union[str,int,None] = None

class Item_delete_phash_features(BaseModel):
    image_id: Union[int ,None] = None
    file_name: Union[None,str] = None

@app.post("/phash_get_similar_images_by_id")
async def phash_get_similar_images_by_id_handler(item: Item_phash_get_similar_images_by_id):
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
        similar = phash_reverse_search(target_features,k,distance_threshold)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
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
        similar = phash_reverse_search(target_features,k,distance_threshold)
        if GET_FILENAMES:
            file_names = get_filenames_bulk([el["image_id"] for el in similar])
            for i in range(len(similar)):
                similar[i]["file_name"] = file_names[i]
        return similar
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error in phash_get_similar_images_by_image_buffer")


@app.post("/calculate_phash_features")
async def calculate_phash_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        image_id = int(image_id)
        if check_if_exists_by_image_id(image_id):
            return Response(content="Image with the same id is already in the db", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, media_type="text/plain")

        features = get_phash(image)
        add_descriptor(image_id,features)
        index.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
        DATA_CHANGED_SINCE_LAST_SAVE = True
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't calculate phash features")

@app.post("/delete_phash_features")
async def delete_phash_features_handler(item: Item_delete_phash_features):
    try:
        global DATA_CHANGED_SINCE_LAST_SAVE
        if item.file_name:
            image_id = get_image_id_by_filename(item.file_name)
        else:
            image_id = item.image_id
        res = index.remove_ids(np.int64([image_id]))
        if res != 0: 
            delete_descriptor_by_id(image_id)
            DATA_CHANGED_SINCE_LAST_SAVE = True
        else: #nothing to delete
            print(f"err: no image with id {image_id}")    
        return Response(status_code=status.HTTP_200_OK)
    except:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Can't delete phash features")

def periodically_save_index(loop):
    global DATA_CHANGED_SINCE_LAST_SAVE, index
    if DATA_CHANGED_SINCE_LAST_SAVE:
        DATA_CHANGED_SINCE_LAST_SAVE=False
        faiss.write_index_binary(index, "./data/populated.index")
    loop.call_later(10, periodically_save_index,loop)

def init_index():
    global index
    if exists("./data/populated.index"):
        index = faiss.read_index_binary("./data/populated.index")
    else:
        print("Index is not found! Exiting...")
        print("Creating empty index")
        import subprocess
        subprocess.call(['python3', 'add_to_index.py'])
        subprocess.call(['python', 'add_to_index.py']) #one shoul exist
        init_index()
main()
