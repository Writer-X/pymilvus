import h5py
import numpy as np
import time 
import random
from bfloat16 import bfloat16

from pymilvus import (
    connections,
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    utility
)

dim = 128
collection_name = "siftBf16"

# configure milvus hostname and port
print(f"\nCreate connection...")
connections.connect(host="localhost", port=19530)

# List all collection names
print(f"\nList collections...")
collection_list = list_collections()
print(list_collections())

if(collection_list.count(collection_name)):
    print(collection_name, " exist, and drop it")
    collection = Collection(collection_name)
    collection.drop()
    print("drop")

field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True)
field2 = FieldSchema(name = "vec", dtype = DataType.BFLOAT16_VECTOR, description = "bfloat16 vector", dim = dim, is_primary = False)
schema = CollectionSchema(fields = [field1, field2], description = "sift decription")
collection = Collection(name = collection_name, data = None, schema = schema, shards_num = 2)

print(list_collections())

print(f"\nList partitions...")
print(collection.partitions)

# insert
fname = "sift-128-euclidean.hdf5"
f = h5py.File(fname, 'r')
data = f['train']
test = f['test']
neighbors = f['neighbors']
print(data.shape)
print(neighbors.shape)
print(test.shape)

print("begin insert...")
counter = 0
block_num = 100
block_size = int(data.shape[0]/block_num)
start = time.time()
for t in range(block_num):
    print("inserting: ", counter, " to ", counter + block_size )
    entities = [
            [i for i in range(counter, counter + block_size)],
            # [vectors[i] for i in range(counter, counter + block_size)]
            [bytes(vec.astype(bfloat16).view(np.uint8).tolist()) for vec in data[counter: counter + block_size]]
            ]
    insert_result =  collection.insert(entities)
    print(insert_result)
    counter = counter + block_size
print ("end of insert, cost: ", time.time()-start)
# flush inside this function
collection.flush()
print(collection.num_entities)

# create index
print(f"\nCreate index...")
# collection.create_index(field_name="vec",
#         index_params={'index_type': 'FLAT',  
#             'metric_type': 'L2',
#             'params': {
#                 'nlist': 128
#                 }})
# collection.create_index(field_name="vec",
#         index_params={'index_type': 'IVF_FLAT',  
#             'metric_type': 'L2',
#             'params': {
#                 'nlist': 128
#                 }})
collection.create_index(field_name="vec",
        index_params={'index_type': 'HNSW',  
            'metric_type': 'L2',
            'params': {
                'M': 4,
                'efConstruction': 100
                }})

# load
print(f"\nLoad...")
collection.load()

# search
print(f"\nSearch...")
res = collection.search([bytes(vec.astype(bfloat16).view(np.uint8).tolist()) for vec in test[0:10]],
                        "vec", {"metric_type": "L2"}, limit=100)
# res = collection.search([vectors[i] for i in range(10)],
#                         "vec", {"metric_type": "L2"}, limit=100)
print(res)

# build run neighbors
run_neighbors = []
for i in range(10):
    run_neighbors.append(res[i].ids)
print("run_neighbors: ", run_neighbors)

def get_recall_values(dataset_neighbors, run_neighbors, nq, count):
    recalls = 0.0
    for i in range(len(run_neighbors[:nq])):
        inter = np.intersect1d(run_neighbors[i][:count], dataset_neighbors[i][:count])
        recalls += inter.shape[0] / count
    return recalls / nq

# recall
print(f"\nRecall...")

recall_value = get_recall_values(neighbors, run_neighbors, 10, 100)
print("recall: ", recall_value)


