import argparse
import torch, os, sys
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
from srdense.proj_utils.local_utils import *
from srdense.thinnet import deepbigdensenet as srnet
from srdense.thinnet import thinnet as srnet

parser = argparse.ArgumentParser(description="PyTorch SRDenseNet Demo")
parser.add_argument("--cuda",  default=True, action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_adam/model_epoch_60.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def batch_forward(cls, BatchData, batch_size):
    total_num = BatchData.shape[0]
    results   = []
    for ind in Indexflow(total_num, batch_size, False):
        data = BatchData[ind]
        data_variable = Variable(torch.from_numpy(data).float(), volatile = True).cuda()
        results.append(cls.forward(data_variable).cpu() )

    return torch.cat(results, dim=0)

def split_testing(cls, img,  batch_size = 4, windowsize=None):
    # since batched_imgs is (C, Row, Col)
    # windowsize = self.row_size
    board = 20
    outputs = np.zeros_like(img) # we don't consider multiple outputs

    PatchDict   =   split_img(img, windowsize = windowsize, board = board, 
                              fixed_window= True, step_size=None)
    print('hahahaha')
    all_keys = PatchDict.keys()
    for this_size in all_keys:
        BatchData, org_slice_list, extract_slice_list, _ = PatchDict[this_size]
        bat, chn, rows, cols = BatchData.shape
        
        thisprediction  =  batch_forward(cls, BatchData, batch_size)
        thisprediction  =  thisprediction.cpu().data.numpy()

        for idx_, _ in enumerate(org_slice_list):
            org_slice     = org_slice_list[idx_]
            extract_slice = extract_slice_list[idx_]
            outputs[: ,org_slice[0], org_slice[1]] = thisprediction[idx_]\
                                                                [:,extract_slice[0], extract_slice[1]]
    return outputs

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = srnet()
model_name = 'thinnet_super_resolution'
model_folder = os.path.join('model_adam', model_name) 

save_folder = 'hd_results'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    
model_path = os.path.join(model_folder, 'model_epoch_{}.pth'.format(91))
model_state = torch.load(model_path)
model.load_state_dict(model_state)
print('reload_model from {}'.format(model_path))


img_root = os.path.join('..','..','Share','Yuanpu_Yun', 'color_neural_results')
gt_root  = os.path.join('data','Parts','HD')

img_name = 'Kidney01-unfiltered.tif'

#im_gt_y  = imread( os.path.join(gt_root, 'kidneytest.bmp')  )
im_l_y   = imread( os.path.join(img_root, img_name)  )
print('Finished loading images')

im_input = im_l_y/127.5 - 1 
im_input = np.transpose(im_input, (2, 0, 1))
print('im_input shape: ', im_input.shape)

#im_input_tensor = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[1], im_input.shape[2])
#im_input = Variable(im_input_tensor, volatile = True)

model = model.cuda()
model.eval()

start_time = time.time()
hr_output = split_testing(model, im_input,  batch_size = 20, windowsize=500)
elapsed_time = time.time() - start_time
print('it takes {} to get the results'.format(elapsed_time))

im_h_y = hr_output.astype(np.float32)
im_h_y = ((im_h_y + 1)*127.5 ).astype(np.uint8)

print('output shape: ', im_h_y.shape)

im_h_y = np.transpose(im_h_y, (1,2,0))

writeImg(im_h_y, os.path.join(save_folder, img_name))

#imshow(im_h_y)

