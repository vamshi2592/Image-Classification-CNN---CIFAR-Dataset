# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:30:42 2017

@author: Matt Green
"""

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile
import helper
import numpy as np
from sklearn import preprocessing


cifar10_dataset_folder_path = 'C://D_Backup/Neural_network_code/image_classification-master/cifar-10-batches-py'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#if not isfile('cifar-10-python.tar.gz'):
 #   with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
  #      urlretrieve(
   #         'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    #        'cifar-10-python.tar.gz',
     #       pbar.hook)

#if not isdir(cifar10_dataset_folder_path):
  #  with tarfile.open('cifar-10-python.tar.gz') as tar:
    #    tar.extractall()
    #    tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)

# %%

# Explore the dataset
batch_id = 2
sample_id = 3
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

# %%

def normalize(x):
    w = 0
    pixels = np.ndarray((len(x), 32, 32, 3))
    for p in x:
        p = p.flatten()
        p = abs((p - 128.) / 128.)
        p = p.reshape(1, 32, 32, 3)
        pixels[w, :, :, :] = p
        w += 1      
    return pixels


tests.test_normalize(normalize)

# %%

def one_hot_encode(x):
    classes = list(range(10))
    lb = preprocessing.LabelBinarizer()
    lb.fit(classes)
    return lb.transform(x)
   
    
tests.test_one_hot_encode(one_hot_encode)

# %%

# Preprocess Training, Validation, and Testing Data
print("Preprocessing and saving data...")
helper.preprocess_and_save_data("cifar-10-batches-py", normalize, one_hot_encode)
