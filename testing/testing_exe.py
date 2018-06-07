import argparse
import torch, os, sys


from torch.autograd import Variable
import numpy as np
import time, math, scipy
import scipy.misc
#import scipy.io as sio

from tifffile import imsave

proj_root = os.path.join('..')
proj_root = os.path.join('..', '..', '..')
sys.path.insert(0, proj_root)

from  srdense.test_utils import split_index, split_testing

parser = argparse.ArgumentParser(description="PyTorch SRDenseNet Demo")
parser.add_argument("--cuda",  default=True, action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_adam/model_epoch_60.pth", type=str, help="model path")
#parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--input",  default = "", help = "input file path")
parser.add_argument("--output", default = "", help = "save path for output")

from srdense.thinnet import tinynet as srnet
model = srnet()
model_name = 'tiny_model'
model_folder = os.path.join('model_adam', model_name) 
load_epoch = 200 #133
    
weights_name = 'model'

#img_root = os.path.join(proj_root, 'data' ,test_folder)
#save_folder = os.path.join(proj_root, 'data','hd_results', test_folder, model_name)
#save_folder = os.path.join('hd_results', model_name)    
#if not os.path.exists(save_folder):
#    os.makedirs(save_folder)

opt = parser.parse_args()
cuda = opt.cuda
use_cuda = cuda and torch.cuda.is_available()

model_path = os.path.join(model_folder, weights_name+'_epoch_{}.pth'.format(load_epoch))
model_state = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(model_state)
print('reload_model from {}'.format(model_path))

img_path  = opt.input
save_path = opt.output

if True:
    if use_cuda:
        model = model.cuda()
    model.eval()

    im_l_y   = scipy.misc.imread( img_path ).astype(np.float16)
    print('Finished loading images')

    im_input = im_l_y/127.5 - 1 
    im_input = np.transpose(im_input, (2, 0, 1)).astype(np.float32)
    print('im_input shape: ', im_input.shape)

    #im_input_tensor = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[1], im_input.shape[2])
    #im_input = Variable(im_input_tensor, volatile = True)

    start_time = time.time()
    hr_output  = split_testing(model, im_input,  batch_size = 2, windowsize=500)
    elapsed_time = time.time() - start_time
    #print('it takes {} to get the results'.format(elapsed_time))

    im_h_y = hr_output.astype(np.float16)
    im_h_y = ((im_h_y + 1)*127.5 ).astype(np.uint8)
    
    print('output shape: ', im_h_y.shape)

    im_h_y = np.transpose(im_h_y, (1,2,0))

    #nake_name = os.path.splitext(img_name)[0]
    #writeImg(im_h_y, os.path.join(save_folder, img_name))
    imsave(save_path, im_h_y, bigtiff=True)
    #imshow(im_h_y)

