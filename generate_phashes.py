from scipy.fft import dct
import numpy as np
from numba import jit
from os import listdir
from joblib import Parallel, delayed
from tqdm import tqdm
import lmdb
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str,nargs='?', default="./../test_images")
parser.add_argument('--use_int_filenames_as_id', type=int, default=1)
args = parser.parse_args()
IMAGE_PATH = args.image_path
USE_INT_FILENAMES = args.use_int_filenames_as_id

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')

def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(4, 'big')
    
DB_filename_to_id = lmdb.open('./filename_to_id.lmdb',map_size=50*1_000_000) #50mb
DB_id_to_filename = lmdb.open('./id_to_filename.lmdb',map_size=50*1_000_000) #50mb

if USE_INT_FILENAMES == 0:
    with DB_id_to_filename.begin(buffers=True) as txn:
        with txn.cursor() as curs:
            curs.last()
            x = curs.item()
            SEQUENTIAL_GLOBAL_ID = int_from_bytes(x[0]) # zeros if id_to_filename.lmdb is empty
    SEQUENTIAL_GLOBAL_ID+=1

DB = lmdb.open('./phashes.lmdb',map_size=500*1_000_000) #500mb

def check_if_exists_by_file_name(file_name):
    if USE_INT_FILENAMES:
        image_id = int(file_name[:file_name.index('.')])
        image_id = int_to_bytes(image_id)
    else:
        with DB_filename_to_id.begin(buffers=True) as txn:
            image_id = txn.get(file_name.encode(), default=False)
            if not image_id:
                return False
    
    with DB.begin(buffers=True) as txn:
        x = txn.get(image_id, default=False)
        if x:
            return True
        return False

@jit(cache=True, nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def fast_phash(image, hash_size=24, highfreq_factor=4): #hash_size=16 for 256bit #hash_size=8 for 64bit
    img_size = hash_size * highfreq_factor
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS) #cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    image = np.array(image)
    dct_data = dct(dct(image, axis=0), axis=1)
    return diff(dct_data, hash_size)

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

def get_phash(query_image):
    bit_list_576 = fast_phash(query_image)
    phash = bit_list_to_72_uint8(bit_list_576)
    return phash

def read_img_file(f):
    img = Image.open(f)
    if img.mode != 'L':
        img = img.convert('L')
    return img

def calc_phash(file_name):
    img_path = IMAGE_PATH+"/"+file_name
    try:
        query_image = read_img_file(img_path)
        phash = get_phash(query_image)
        return [file_name, phash.tobytes()]
    except:
        print("error")
        print(file_name)
        return None

file_names = listdir(IMAGE_PATH)
print(f"images in {IMAGE_PATH} = {len(file_names)}")

new_images = []
for file_name in tqdm(file_names):
    if check_if_exists_by_file_name(file_name):
        continue
    new_images.append(file_name)

print(f"new images = {len(new_images)}")
new_images = [new_images[i:i + 100000] for i in range(0, len(new_images), 100000)]
for batch in new_images:
    phashes = Parallel(n_jobs=-1, verbose=1)(delayed(calc_phash)(file_name) for file_name in batch)
    phashes = [i for i in phashes if i]  # remove None's
    file_name_to_id = []
    id_to_file_name = []    
    for i in range(len(phashes)):
        if USE_INT_FILENAMES:
            idx_of_dot = phashes[i][0].index('.')
            image_id = int_to_bytes(int(phashes[i][0][:idx_of_dot]))
        else:
            image_id = int_to_bytes(SEQUENTIAL_GLOBAL_ID)
            SEQUENTIAL_GLOBAL_ID+=1

        file_name = phashes[i][0].encode()
        phashes[i][0] = image_id
        phashes[i] = tuple(phashes[i])
        file_name_to_id.append((file_name, image_id))
        id_to_file_name.append((image_id, file_name))
        
    with DB_filename_to_id.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(file_name_to_id)

    with DB_id_to_filename.begin(write=True, buffers=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(id_to_file_name)

    print("pushing data to db")
    with DB.begin(write=True, buffers=True) as txn:
        with txn.cursor() as curs:
            curs.putmulti(phashes)
