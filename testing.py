import argparse
import torch, os, sys
from torch.autograd import Variable
import numpy as np
import time, math, scipy
import scipy.io as sio
#import matplotlib.pyplot as plt
#from srdense.proj_utils.local_utils import *
from srdense.test_utils import *

def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    rmgimg = scipy.misc.imread(imgfile)
    return rmgimg

def writeImg(array, savepath):
    scipy.misc.imsave(savepath, array)


weights_name = 'model'
if 0:
    from srdense.thinnet import thinnet as srnet
    model = srnet()
    model_name = 'thinnet_super_resolution'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 91

if 0:
    from srdense.thinnet import deepbigdensenet as srnet
    model = srnet()
    model_name = 'bigdeepdensenet_super_resolution'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 150

if 1:
    from srdense.thinnet import tinynet as srnet
    model = srnet()
    model_name = 'tiny_model'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 104 #133
    
if 1: 
    from srdense.thinnet import tinynet as srnet
    model = srnet()
    model_name = 'tiny_patch_model'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 135

if 0:
    from srdense.thinnet import deepbigdensenet as srnet
    model = srnet()
    model_name = 'bigdeep_patch_model'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 150

if 0:
    from srdense.cyclenet import Generator as srnet
    model = srnet()
    model_name = 'cycle_super'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 68
    weights_name = 'netG_A2B_model'

if 0:
    from srdense.cyclenet import giantnet as srnet
    model = srnet()
    model_name = 'giant_patch_model'
    model_folder = os.path.join('model_adam', model_name) 
    load_epoch = 68

parser = argparse.ArgumentParser(description="PyTorch SRDenseNet Demo")
parser.add_argument("--cuda",  default=True, action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_adam/model_epoch_60.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

opt = parser.parse_args()
cuda = opt.cuda
test_folder = 'FInalData'

img_root = os.path.join('..','..','Share','Yuanpu_Yun', 'color_neural_results')
img_list = ('Kidney01-unfiltered.tif', 'Onion01-unfiltered.tif' , 
            'Basswood01-unfiltered.tif', 'Tonsil01-unfiltered.tif'
            )

img_root = os.path.join('data', test_folder, 'color_neural_results')
img_list = ( 'Kidney01-unfiltered.bmp', 'Onion01-unfiltered.bmp' , 
            'SpinalCord01-unfiltered.bmp', 'Tonsil01-unfiltered.bmp'
            )
#img_name  = 'Kidney01-unfiltered.tif'
#img_name = 'Onion01-unfiltered.tif'
#img_name = 'Basswood01-unfiltered.tif'
#img_name = 'Tonsil01-unfiltered.tif'
#img_name = 'kidney_part.bmp'
#img_name  =  'resized_hd_stone.tif'
#im_gt_y  = imread( os.path.join(gt_root, 'kidneytest.bmp')  )

for img_name in img_list:
    save_folder = os.path.join('hd_results', model_name, test_folder)
    #save_folder = os.path.join('hd_results', model_name)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    model_path = os.path.join(model_folder, weights_name+'_epoch_{}.pth'.format(load_epoch))
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    print('reload_model from {}'.format(model_path))

    model = model.cuda()
    model.eval()

    im_l_y   = imread( os.path.join(img_root, img_name)  ).astype(np.float16)
    print('Finished loading images')

    im_input = im_l_y/127.5 - 1 
    im_input = np.transpose(im_input, (2, 0, 1)).astype(np.float32)
    print('im_input shape: ', im_input.shape)

    #im_input_tensor = torch.from_numpy(im_input).float().view(1, -1, im_input.shape[1], im_input.shape[2])
    #im_input = Variable(im_input_tensor, volatile = True)

    start_time = time.time()
    hr_output = split_testing(model, im_input,  batch_size = 2, windowsize=500)
    elapsed_time = time.time() - start_time
    print('it takes {} to get the results'.format(elapsed_time))

    im_h_y = hr_output.astype(np.float16)
    im_h_y = ((im_h_y + 1)*127.5 ).astype(np.uint8)
    
    print('output shape: ', im_h_y.shape)

    im_h_y = np.transpose(im_h_y, (1,2,0))

    writeImg(im_h_y, os.path.join(save_folder, img_name))

    #imshow(im_h_y)

