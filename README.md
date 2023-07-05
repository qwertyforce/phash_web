# phash_web
Faiss + Numba + SciPy + FastAPI + LMDB <br>
Uses DCT-based phash (576 bit) <br>
Supported operations: add new image, delete image, find similar images by image file, find similar images by image id

```pip3 install -r requirements.txt```

```generate_phashes.py ./path_to_img_folder``` -> generates features  
```--use_int_filenames_as_id=0``` - images get sequential ids  
```--use_int_filenames_as_id=1``` - image id is parsed from filename ("123.jpg" -> 123)

```add_to_index.py``` -> adds features from lmdb to Flat index  

```phash_web.py``` -> web microservice  
```GET_FILENAMES=1 phash_web.py``` -> when searching, include filename in search results  

DOCKER:  
build image - ```docker build -t qwertyforce/phash_web:1.0.0 --network host -t qwertyforce/phash_web:latest ./```  
  
run interactively - ```docker run -ti --rm -p 127.0.0.1:33336:33336  --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name phash_web qwertyforce/phash_web:1.0.0```  
  
run as deamon - ```docker run -d --rm -p 127.0.0.1:33336:33336  --network=ambience_net --mount type=bind,source="$(pwd)"/data,target=/app/data --name phash_web qwertyforce/phash_web:1.0.0 ```  

