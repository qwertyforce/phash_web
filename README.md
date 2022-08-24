# phash_web
Faiss + Numba + SciPy + FastAPI + LMDB <br>
Uses DCT-based phash (576 bit) <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id

```pip3 install -r requirements.txt```

```generate_phashes.py ./path_to_img_folder``` -> generates features  
```add_to_index.py``` -> adds features from lmdb to Flat index  
```phash_web.py``` -> web microservice  
