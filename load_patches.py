import torch
import h5py
from torch.multiprocessing import Pool
from scipy import ndimage
import scipy, skimage
from srdense.proj_utils.local_utils import *


f = open('file_name.ext', 'r')
x = f.readlines()
f.close()
import sys, os

file_path = os.path.join()

with open(file_path) as text_file:
    lines = text_file.readlines().split(',')