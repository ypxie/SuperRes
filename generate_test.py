import torch.utils.data as data
import torch, os, sys
import h5py
from torch.multiprocessing import Pool

from srdense.proj_utils.local_utils import *

this_down_ratio = 0.5
this_hd_path = os.path.join('data', 'HD', 'stone.tif')
save_path = os.path.join('..','..','Share','Yuanpu_Yun', 'color_neural_results', 'resized_hd_stone.tif')

hd_img_orig = imread(this_hd_path)
hd_row, hd_col, chn = hd_img_orig.shape

hd_row, hd_col, chn = hd_img_orig.shape
reg_row, reg_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)

reg_img_down = imresize_shape(hd_img_orig, (reg_row, reg_col) )
reg_img_orig = imresize_shape(reg_img_down, (hd_row, hd_col) )

writeImg(reg_img_orig, save_path)