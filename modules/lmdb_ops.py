import lmdb

DB_phash = lmdb.open('./data/phashes.lmdb',map_size=500*1_000_000) #500mb
DB_filename_to_id = lmdb.open('./data/filename_to_id.lmdb',map_size=50*1_000_000) #50mb
DB_id_to_filename = lmdb.open('./data/id_to_filename.lmdb',map_size=50*1_000_000) #50mb

def get_dbs():
    return DB_phash, DB_filename_to_id, DB_id_to_filename
