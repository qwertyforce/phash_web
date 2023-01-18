from scipy.fft import dct
import numpy as np
from numba import jit
from PIL import Image

@jit(cache=True, nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()

def fast_phash(image, hash_size): #hash_size=16 for 256bit #hash_size=8 for 64bit
    
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

def get_phash_by_image(image, hash_size=24, highfreq_factor=4, aug=False):
    img_size = hash_size * highfreq_factor
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS) #cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    image = np.array(image)

    bit_list_576 = fast_phash(image, hash_size)
    phash = bit_list_to_72_uint8(bit_list_576)
    if aug:
        mirrored_query_image = np.fliplr(image)
        bit_list_576_mirrored = fast_phash(mirrored_query_image, hash_size)
        phash_mirrored = bit_list_to_72_uint8(bit_list_576_mirrored)
        return np.array([phash, phash_mirrored])
    return phash