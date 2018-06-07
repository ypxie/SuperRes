import torch, os, sys
from torch.autograd import Variable
import numpy as np
import time, math
#import scipy.io as sio
#import matplotlib.pyplot as plt
#from .proj_utils.local_utils import *

def split_index(rowsize, colsize, windowsize=1000, board = 0, fixed_window = False, step_size = None):
    
    IndexDict = {}
    identifier = -1
    PackList = []
    if windowsize is not None and  type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    if windowsize is None or (rowsize <= windowsize[0] and colsize<=windowsize[1] ):

        place_slice  = (slice(0, rowsize), slice(0, colsize))
        output_slice = place_slice
        crop_patch_slice = (slice(0, rowsize), slice(0, colsize))
        thisSize =  (rowsize, colsize )
        identifier = identifier + 1

        if thisSize in IndexDict:
           IndexDict[thisSize].append(identifier)
        else:
           IndexDict[thisSize] = []
           IndexDict[thisSize].append(identifier)
        #PackList.append((crop_patch_slice, place_slice, output_slice, thisSize, identifier))
        PackList.append((crop_patch_slice, place_slice, output_slice, thisSize))
    else:

        hidden_windowsize = (windowsize[0]-2*board, windowsize[1]-2*board)
        
        if type(step_size) is int:
            step_size = (step_size, step_size)
        if step_size is None:
            step_size = hidden_windowsize

        numRowblocks = int(math.ceil(float(rowsize)/hidden_windowsize[0]))  # how many windows we need
        numColblocks = int(math.ceil(float(colsize)/hidden_windowsize[1]))
        
        # sanity check, make sure the image is at least of size window_size to the left-hand side if fixed_windows is true
        # which means,    -----*******|-----, left to the vertical board of original image is at least window_size.

        thisrowstart, thiscolstart =0, 0
        thisrowend,   thiscolend = 0,0
        
        for row_idx in range(numRowblocks):
            thisrowlen   = min(hidden_windowsize[0], rowsize - row_idx * step_size[0])
            # special case for the first row and column. 

            thisrowstart = 0 if row_idx == 0 else thisrowstart + step_size[0]

            thisrowend = thisrowstart + thisrowlen

            row_shift = 0
            if fixed_window:
                #if thisrowlen < hidden_windowsize[0]:
                #    row_shift = hidden_windowsize[0] - thisrowlen
                
                if thisrowend + board >= rowsize:  # if we can not even pad it. 
                    row_shift = (hidden_windowsize[0] - thisrowlen) + (thisrowend+board - rowsize)
                    
            for col_idx in range(numColblocks):
                #import pdb; pdb.set_trace()
                thiscollen = min(hidden_windowsize[1], colsize -  col_idx * step_size[1])
 
                thiscolstart = 0 if col_idx == 0 else thiscolstart + step_size[1]

                thiscolend = thiscolstart + thiscollen

                col_shift = 0
                if fixed_window:
                    # we need to shift the patch to left to make it at least windowsize.
                    #if thiscollen < hidden_windowsize[1]:
                    #    col_shift = hidden_windowsize[1] - thiscollen

                    if thiscolend + board >= colsize:  # if we can not even pad it. 
                        col_shift = (hidden_windowsize[1] - thiscollen) + (thiscolend+board - colsize)

                #
                #----board----******************----board----
                #

                if thisrowstart == 0:
                    #------ slice obj for crop from original image-----
                    crop_r_start = thisrowstart
                    crop_r_end  =  thisrowend + 2*board 

                    #------ slice obj for crop results from local patch-----
                    output_slice_row_start = 0
                    output_slice_row_end   = output_slice_row_start + thisrowlen 
                else:
                    #------ slice obj for crop from original image-----
                    crop_r_start = thisrowstart - board - row_shift 
                    crop_r_end  =  min(rowsize, thisrowend   + board)
                    
                    #------ slice obj for crop results from local patch-----
                    output_slice_row_start = board + row_shift
                    output_slice_row_end   = output_slice_row_start + thisrowlen
                
                if thiscolstart == 0:
                    #------ slice obj for crop from original image-----
                    crop_c_start = thiscolstart
                    crop_c_end  =  thiscolend + 2* board 
                    #------ slice obj for crop results from local patch-----
                    output_slice_col_start = 0
                    output_slice_col_end   = output_slice_col_start + thiscollen
                else:
                    #------ slice obj for crop from original image-----
                    crop_c_start =  thiscolstart - board - col_shift 
                    crop_c_end   =  min(colsize, thiscolend  + board) 
                    #------ slice obj for crop results from local patch-----
                    output_slice_col_start = board + col_shift
                    output_slice_col_end   = output_slice_col_start + thiscollen

                # slice on a cooridinate of the original image for the central part
                place_slice  = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))
                
                # extract on local coordinate of a patch to fill place_slice
                output_slice = ( slice(output_slice_row_start, output_slice_row_end),
                                 slice(output_slice_col_start, output_slice_col_end))

                # here take care of the original image size
                crop_patch_slice = (slice(crop_r_start, crop_r_end), slice(crop_c_start, crop_c_end))
                thisSize = (thisrowlen + 2*board + row_shift, thiscollen + 2*board + col_shift)
                
                
                identifier =  identifier +1
                #PackList.append((crop_patch_slice, place_slice, output_slice, thisSize, identifier))
                PackList.append((crop_patch_slice, place_slice, output_slice, thisSize))
                if thisSize in IndexDict:
                   IndexDict[thisSize].append(identifier)
                else:
                   IndexDict[thisSize] = []
                   IndexDict[thisSize].append(identifier)
                #print('this crop and place: ', (crop_patch_slice, place_slice, output_slice, thisSize))

    return  PackList


def split_testing(cls, imgs,  batch_size = 4, windowsize=None):
    # since batched_imgs is (B, C, Row, Col)
    # windowsize = self.row_size
    board = 20
    adptive_batch_size=False # cause we dont need it for fixed windowsize.
    chn, row_size, col_size = imgs.shape
    assert len(imgs.shape) == 3, 'wrong input dimensio of input image'
    outputs = np.zeros_like(imgs, dtype=np.float32) # we don't consider multiple outputs
    print('Finish allocate outputs')
    
    PackList   =   split_index(row_size, col_size, windowsize = windowsize, board = board, fixed_window= True, step_size=None)
    print('Finished splitting')
    total_testing = len(PackList)
    _,_,_, thisSize = PackList[0]

    BatchData = np.zeros((batch_size, chn, thisSize[0], thisSize[1]) , dtype=np.float32)

    batch_count = 0
    results = []
    for idx in range(total_testing):
        crop_patch_slice, place_slice, output_slice, _ = PackList[idx]
        BatchData[batch_count] = imgs[:,crop_patch_slice[0],crop_patch_slice[1]]
        batch_count += 1
        if batch_count == batch_size or idx == total_testing - 1:
            data = BatchData[0:batch_count]
            with torch.no_grad():
                data_variable = Variable(torch.from_numpy(data).float())
                if cls.parameters().__next__().is_cuda:
                    data_variable = data_variable.cuda(cls.parameters().__next__().get_device())
                
            results.append(cls.forward(data_variable).cpu().data.numpy() )
            batch_count = 0

    idx = 0
    for each_result in results:
        for b_idx in range(each_result.shape[0]):
            _, place_slice, output_slice, _ = PackList[idx]
            outputs[:, place_slice[0], place_slice[1]] = each_result[b_idx, :,output_slice[0], output_slice[1]]
            idx += 1
    #results = torch.cat(results, dim = 0).numpy()
    #for idx in range(total_testing):
    #    _, place_slice, output_slice, _ = PackList[idx]
        #import pdb; pdb.set_trace()
    #    print(place_slice, output_slice, results[idx].shape, results[idx][:,output_slice[0], output_slice[1]].shape)
    #    outputs[:, place_slice[0], place_slice[1]] = results[idx][:,output_slice[0], output_slice[1]]

    return outputs

#------------