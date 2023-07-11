import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

tensor_bin_aipp = np.random.rand(1,3,256,256).astype(np.uint8)
tensor_bin_nor = np.random.rand(1,3,224,224).astype(np.float32)
tensor_npy_aipp = np.random.rand(1,3,256,256).astype(np.uint8)
tensor_npy_nor = np.random.rand(1,3,224,224).astype(np.float32)
data_num = 8
list_k = list(range(data_num))
base_path = os.getcwd()
cur_bin_aipp_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_bin_aipp")
if os.path.exists(cur_bin_aipp_path):
    shutil.rmtree(cur_bin_aipp_path)
os.makedirs(cur_bin_aipp_path)
cur_bin_nor_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_bin_nor")
if os.path.exists(cur_bin_nor_path):
    shutil.rmtree(cur_bin_nor_path)
os.makedirs(cur_bin_nor_path)
cur_npy_aipp_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_npy_aipp")
if os.path.exists(cur_npy_aipp_path):
    shutil.rmtree(cur_npy_aipp_path)
os.makedirs(cur_npy_aipp_path)
cur_npy_nor_path = os.path.join(base_path, "testdata/resnet50/input/fake_dataset_npy_nor")
if os.path.exists(cur_npy_nor_path):
    shutil.rmtree(cur_npy_nor_path)
os.makedirs(cur_npy_aipp_path)
for i,_ in enumerate(tqdm(list_k, file=sys.stdout, desc='generate dataset process:')):
    bin_aipp_name = f"{i}.bin"
    bin_nor_name = f"{i}.bin"
    npy_aipp_name = f"{i}.npy"
    npy_nor_name = f"{i}.npy"
    bin_aipp_path = os.path.join(cur_bin_aipp_path, bin_aipp_name)
    bin_nor_path = os.path.join(cur_bin_nor_path, bin_nor_name)
    npy_aipp_path = os.path.join(cur_npy_aipp_path, npy_aipp_name)
    npy_nor_path = os.path.join(cur_npy_nor_path, npy_nor_name)
    tensor_bin_aipp.tofile(bin_aipp_path)
    tensor_bin_nor.tofile(bin_nor_path)
    np.save(npy_aipp_path, tensor_npy_aipp)
    np.save(npy_nor_path, tensor_npy_nor)