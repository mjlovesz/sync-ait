import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

tensor_bin = np.random.rand(1,3,256,256).astype(np.uint8)
tensor_npy = np.random.rand(1,3,256,256).astype(np.uint8)
data_num = 8
list_k = list(range(data_num))
base_path = os.getcwd()
cur_bin_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_bin")
if os.path.exists(cur_bin_path):
    shutil.rmtree(cur_bin_path)
os.makedirs(cur_bin_path)
cur_npy_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_npy")
if os.path.exists(cur_npy_path):
    shutil.rmtree(cur_npy_path)
os.makedirs(cur_npy_path)
for i,_ in enumerate(tqdm(list_k, file=sys.stdout, desc='generate dataset process:')):
    bin_name = f"{i}.bin"
    npy_name = f"{i}.npy"
    bin_path = os.path.join(cur_bin_path, bin_name)
    npy_path = os.path.join(cur_npy_path, npy_name)
    tensor_bin.tofile(bin_path)
    np.save(npy_path, tensor_npy)