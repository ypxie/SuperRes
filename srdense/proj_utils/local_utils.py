# -*- coding: utf-8 -*-

import numpy as np
import os, math
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import skimage, skimage.morphology

from PIL import Image, ImageDraw
from scipy.ndimage.interpolation import rotate
from skimage import color, measure
import re
import scipy.ndimage

from numba import jit, autojit

import random, shutil
import scipy.misc as misc

def get(identifier):
    return get_from_module(identifier, globals(), 'local_utils')


def mkdirs(folders, erase=False):
    if type(folders) is not list:
        folders = [folders]
    for fold in folders:
        if not os.path.exists(fold):
            os.makedirs(fold)
        else:
            if erase:
                shutil.rmtree(fold)
                os.makedirs(fold)


class myobj(object):
    pass


def process_sent(this_sent):
    this_sent = ' <start> ' + this_sent + ' <eos> '
    return this_sent

def split_words(words):
    words = words.replace('_', ' ')
    return re.findall(r'\w+|\S+', words)

def auc(x, y, reorder=False):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.
    Parameters
    ----------
    x : array, shape = [n]
        x coordinates.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.
    Returns
    -------
    auc : float
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    See also
    --------
    roc_auc_score : Computes the area under the ROC curve
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    #check_consistent_length(x, y)
    #x = column_or_1d(x)
    #y = column_or_1d(y)

    x, y = np.asarray(x), np.asarray(y)
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    if reorder:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_)/ (max_ - min_ + 1e-9)
    X = X*255
    return X.astype(np.uint8)

def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    rmgimg = scipy.misc.imread(imgfile)
    return rmgimg


def writeImg(array, savepath):
    scipy.misc.imsave(savepath, array)
    #cv2.imwrite(savepath,  array)


def imresize(img, resizeratio=1):
    '''Take care of cv2 reshape squeeze behevaior'''
    if resizeratio == 1:
        return img
    #outshape = ( int(img.shape[1] * resizeratio) , int(img.shape[0] * resizeratio)) 
    #temp = cv2.resize(img, outshape).astype(float)
    outshape = ( int(img.shape[0] * resizeratio) , int(img.shape[1] * resizeratio)) 
    temp = misc.imresize(img, size=outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imresize_shape(img, outshape):
    if len(img.shape) == 4:
        img_out = []
        for this_img in img:
            img_out.append( imresize_shape(this_img, outshape) )
        return np.stack(img_out, axis=0)

    img = img.astype(np.float32)
    outshape = ( int(outshape[0]) , int(outshape[1])  )
    if (img.shape[0], img.shape[1]) == outshape:
        return img
    #temp = cv2.resize(img, (outshape[1], outshape[0]))
    temp = misc.imresize(img, size=outshape, interp='bilinear').astype(float)

    return temp

def pre_process_img(img, yuv = False, mode = 'vanilla', norm = True):
    if yuv :
      img = RGB2YUV(img)
    img = img.astype('float32')
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis = -1)
    # first of all normalize the img
    if norm:
        if mode == 'channel':
            for ch_id in range(img.shape[2]):
                this_ch = img[:,:, ch_id]
                this_mean =  np.mean(this_ch)
                this_std  =  np.std(this_ch)
                img[:,:, ch_id] = (this_ch - this_mean) / (this_std + 1e-8)
        elif mode == 'whole':
            this_mean, this_std = np.mean(img[:]), np.std(img[:])
            img = (img - this_mean)/this_std

        elif mode == 'rescale':
            this_min, this_max, this_mean = np.min(img[:]), np.max(img[:]),np.mean(img[:])
            img = (img - this_min) / ( this_max- this_min + 1e-8)
            img = img - np.mean(img)

        elif mode == 'vanilla':
            img = img.astype(np.float32)/255
            img = (img - 0.5)/0.5
        else:
            raise Exception('Unknown mode for pre_processing')
    return img

def mysqueeze(a, axis = None):
    if axis == None:
        return np.squeeze(a)
    if a.shape[axis] != 1:
        return a
    else:
        return np.squeeze(a, axis = axis)

def getImg_from_Grid(grid_vec, patchsize):
    patchRow, patchCol = patchsize
    indx =  -1
    imgchannel = int(grid_vec.shape[-1]//(patchRow*patchCol))
    numberofImg = grid_vec.shape[0]
    gridshape = (grid_vec[0,:,:,:].shape[0],grid_vec[0,:,:,:].shape[1])
    imgs = np.zeros((grid_vec.shape[0], gridshape[0]*patchRow, gridshape[1]*patchCol, imgchannel ))
    imgs = mysqueeze(imgs, axis = -1)

    for imgidx  in range(numberofImg):
        for colid in range(gridshape[1]):
           for rowid in range(gridshape[0]):
              indx = indx + 1
              this_vec =  grid_vec[imgidx,rowid,colid,:]
              this_patch = np.reshape(this_vec, (patchRow,patchCol,imgchannel ))
              this_patch = mysqueeze(this_patch,axis = -1)
              startRow, endRow = rowid *patchRow, (rowid+1)*patchRow
              startCol, endCol = colid *patchCol, (colid+1)*patchCol
              #print this_patch.shape
              imgs[imgidx,startRow:endRow,startCol: endCol] = this_patch
              #imshow(img)
    return imgs

def getmesh_zigzag(RowPts,ColPts):
    """RowPts means the index of row coordinates,
       ColPts means the index of col coordinates
    """
    #img = np.zeros((max(RowPts), max(ColPts)))
    rr,cc = [], []
    for colidx in ColPts:
        if np.mod(colidx, 2) == 0:
           rr.extend(RowPts)
        else:
           rr.extend(RowPts[::-1])
        cc.extend([colidx]*len(RowPts))

    return np.asarray(rr), np.asarray(cc)

def getmesh(RowPts,ColPts):
    """RowPts means the index of row coordinates,
       ColPts means the index of col coordinates
    """
    rr,cc = [], []
    for colidx in ColPts:
        rr.extend(RowPts)
        cc.extend([colidx]*len(RowPts))
    return np.asarray(rr), np.asarray(cc)

def getfileinfo(imgdir, contourextList, ImgExtList, LabelExt, test_mode = False):
    '''return a list of dictionary {'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile}
    '''
    alllist  = [f for f in os.listdir(imgdir)]
    alllist = sorted(alllist)
    
    returnList = []
    for f in alllist:
        if os.path.isfile(os.path.join(imgdir,f)) and \
                   os.path.splitext(f)[1] in ImgExtList:
            
            if test_mode is False:
                flag = 0
                for contourext in contourextList:
                    thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + contourext + LabelExt)
                    if os.path.isfile(thismatfile):
                        this_dict = {'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile}
                        this_dict['file_name'] = f
                        this_dict['mat_name'] = os.path.splitext(f)[0] + contourext + LabelExt
                        returnList.append(this_dict)

                        flag = 1
                        break
                if flag == 0:
                    print(("Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )))
            else:
                
                this_dict = {'thisfile':os.path.join(imgdir,f), 'thismatfile':None}
                this_dict['file_name'] = f
                this_dict['mat_name'] = None
                returnList.append(this_dict)
    return returnList



def yieldfileinfo(imgdir, contourextList,ImgExtList,LabelExt):
    alllist  = [f for f in os.listdir(imgdir)]
    alllist = sorted(alllist)
    #absfilelist = [];
    #absmatfilelist = [];
    dict_lsit = []
    returnList = []
    for f in alllist:
        if  os.path.isfile(os.path.join(imgdir,f)) and \
            os.path.splitext(f)[1] in ImgExtList:
            flag = 0
            returnDict = {}
            for contourext in contourextList:
                thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + contourext + LabelExt)
                if os.path.isfile(thismatfile):
                    #absmatfilelist.append(thismatfile)
                    #absfilelist.append(os.path.join(imgdir,f))
                    returnDict['thisfile']    = os.path.join(imgdir,f)
                    returnDict['thismatfile'] = thismatfile
                    #returnList.append({'thisfile':os.path.join(imgdir,f), 'thismatfile':thismatfile})
                    dict_lsit.append(returnDict)
                    flag = 1
                    break
            if  flag == 0:
                print(("Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )))
    return dict_lsit


def getFromFolderList(subfolder_list,  number_list = -1, contourextList = '',
               ImgExtList = '.png',LabelExt = '.mat'):
    '''
    subfolder_list: the folder that contain the images,  it is a list of folders.
    number_list: the number of images you wanna take
    '''
    random.seed(4)
    if type(subfolder_list) != list:
        subfolder_list = [subfolder_list]
    if type(number_list) != list:
        number_list = [number_list]
    if len(number_list) == 1:
        number_list = number_list * len(subfolder_list)

    returnDict_list = []
    for imgdir, num_img in zip(subfolder_list, number_list):
        alllist  = [f for f in os.listdir(imgdir)]
        if len(subfolder_list) == 1 and len(alllist) < num_img:
            return None
            # because we want to terminate when the number of image is larger than this.

        total_num = len(alllist)
        list_of_file = list(range(total_num))
        random.shuffle(list_of_file)
        img_count = 0
        for file_ind in list_of_file:
            returnDict = {}
            f = alllist[file_ind]
            if os.path.isfile(os.path.join(imgdir,f)) and \
                    os.path.splitext(f)[1] in ImgExtList:
                flag = 0
                for contourext in contourextList:
                    thismatfile  = os.path.join(imgdir,os.path.splitext(f)[0] + \
                    contourext + LabelExt)
                    if os.path.isfile(thismatfile):
                        returnDict['thisfile'] = os.path.join(imgdir,f)
                        returnDict['thismatfile'] = thismatfile
                        returnDict_list.append(returnDict)
                        flag = 1
                        img_count += 1
                        break
                if flag == 0:
                    print(("Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )))
            if num_img > 0 and img_count == num_img:
                break
    return  returnDict_list

def getfilelist(Imagefolder, inputext, with_ext=False):
    '''inputext: ['.json'] '''
    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    allfiles = sorted(os.listdir(Imagefolder))
    for f in allfiles:
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
                filelist.append(os.path.join(Imagefolder,f))
                if with_ext is True:
                    filenames.append(  os.path.basename(f)   )
                else:
                    filenames.append(  os.path.splitext(os.path.basename(f))[0]   )
    return filelist, filenames

def getfolderlist(Imagefolder):
    '''inputext: ['.json'] '''
    folder_list = []
    folder_names = []
    allfiles = sorted(os.listdir(Imagefolder))

    for f in allfiles:
        this_path = os.path.join(Imagefolder, f)
        if os.path.isdir(this_path):
            folder_list.append(this_path)
            folder_names.append(f)
    return folder_list, folder_names

def find(logicalMatrix):
    totalInd = np.arange(0, len(logicalMatrix.flat))
    return totalInd[logicalMatrix.flatten()]

def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()

def fast_Points2Patches(Patches,centerIndx, img, patchsize):
    totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
    numberofInd = len(centerIndx)
    #Patches = np.zeros(numberofInd, np.prod(patchsize)*img.shape[2])
    if len(img.shape) == 2:
        img = img[:,:,None]
    npad3 = ((patchsize[0],patchsize[0]),(patchsize[1],patchsize[1]),(0,0))
    img = np.pad(img,npad3, 'symmetric')
    centralRow = totalsub[0][:] + patchsize[0]
    centralCol = totalsub[1][:] + patchsize[1]

    se = CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

    for i in range(numberofInd):
        Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].copy().flatten()


def knnsearch(seeds, pints,K):
    """return the indexes and distance of k neareast points for every pts in points from seeds\
    seeds: N*dim, points: N*dim
    seeds, and points should be of N*dim format"""
    knn = NearestNeighbors(n_neighbors=K)
    knn.fit(seeds)
    distance, index  = knn.kneighbors(pints, return_distance=True)
    return index,distance

def Points2Patches(centerIndx, img, patchsize):
    totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
    numberofInd = len(centerIndx)
    if len(img.shape) == 2:
        img = img[:,:,None]
    Patches = np.zeros((numberofInd, np.prod(patchsize)*img.shape[2]))
    npad3 = ((patchsize[0],patchsize[0]),(patchsize[1],patchsize[1]),(0,0))
    img = np.pad(img,npad3, 'symmetric')
    centralRow = totalsub[0][:] + patchsize[0]
    centralCol = totalsub[1][:] + patchsize[1]

    se = CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

    for i in range(numberofInd):
       Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].copy().flatten()
            #imshow(img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:][...,0])
        #       tmp = img[:,:,0].copy() #np.zeros((img.shape[0], img.shape[1]))
        #       tmp[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i]] = 255
        #       #tmp = scipy.ndimage.morphology.grey_dilation(tmp,(3,3) )
        #       imshow(tmp)
    return Patches


def CentralToOrigin(centralRow, centralCol,Rowsize,Colsize):

    RowUp = int(Rowsize/2)
    RowDown = Rowsize - RowUp - 1
    ColLeft = int(Colsize/2)
    ColRight = Colsize - ColLeft - 1
    se = {}
    se['RS'] = int(centralRow - RowUp)
    se['RE'] = int(centralRow + RowDown + 1)  #because python does not take the last value
    se['CS'] = int(centralCol - ColLeft)
    se['CE'] = int(centralCol + ColRight + 1)
    return se

def OriginToCentral(OrigRow, OrigCol,Rowsize,Colsize):
    RowUp = int(Rowsize/2)
    ColLeft = int(Colsize/2)
    center = {}
    center['RC'] = int(OrigRow + RowUp)
    center['CC'] = int(OrigCol + ColLeft)
    return center


def patchflow(Img,chunknum,row,col,channel,**kwargs):

    pixelind = find(np.ones(Img.shape[0], Img.shape[1]) == 1)
    Totalnum = len(pixelind)
    numberofchunk = np.floor((Totalnum + chunknum - 1)// chunknum)   # the floor
    Chunkfile = np.zeros((chunknum, row*col*channel))

    chunkstart = 0
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = pixelind[chunkstart: chunkstart + thisnum]
        fast_Points2Patches(Chunkfile[0:thisnum,:],thisInd, Img, (row,col))
        chunkstart += thisnum
        yield Chunkfile[0:thisnum,:]

def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd

def batchflow(batch_size, *Data):
    # we dont check Data, they should all have equal first dimension
    Totalnum = Data[0].shape[0]
    for thisInd in Indexflow(Totalnum, batch_size):
        if len(Data) == 1:
            yield Data[0][thisInd, ...]
        else:
            batch_tuple = [s[thisInd,...] for s in Data]
            yield tuple(batch_tuple)

@autojit
def overlayImg(img, mask,print_color =[5,119,72],linewidth= 1, alpha = 0.618,savepath = None):
    #img = img_as_float(data.camera())
    rows, cols = img.shape[0:2]
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    assert len(mask.shape) == 2,'mask should be of dimension 2'
    color_mask[mask == 1] = print_color
    color_mask[mask == 0] = img[mask == 0]
    #imshow(color_mask)

    if len(img.shape) == 2:
       img_color = np.dstack((img, img, img))
    else:
       img_color = img

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    # Display the output
    #f, (ax0, ax1, ax2) = plt.subplots(1, 3,
    #                                  subplot_kw={'xticks': [], 'yticks': []})
    #ax0.imshow(img, cmap=plt.cm.gray)
    #ax1.imshow(color_mask)
    #ax2.imshow(img_masked)
    #plt.show()
    img_masked = np.asarray((img_masked/np.max(img_masked) ) * 255, dtype = np.uint8)

    if savepath is not None:
        im = Image.fromarray(img_masked)
        im.save(savepath)

    #im = Image.fromarray(img_masked)
    #im.save(savepath)
    return img_masked

@jit
def _combine_markers(label_img, coordinates):
    #first we remove all the label_img region contain coordinates
    tmp_img = label_img.copy()
    num_obj = np.max(tmp_img)

    for ind in range(1, num_obj+1):
        for j in range(coordinates.shape[0]):
            if tmp_img[coordinates[j,0], coordinates[j,1]] == ind:
                tmp_img[tmp_img==ind] = 0
                break
    new_num = np.max(tmp_img)
    rest_contour = label2contour(tmp_img, returnImg =False)
    old_num = coordinates.shape[0]
    total_num = len(rest_contour) + old_num
    new_coord = np.zeros((total_num, 2))
    new_coord[0:old_num] = coordinates
    for ind, this_contour in enumerate(rest_contour):
        new_coord[old_num+ind] = np.asarray([np.mean(this_contour[:,0]), np.mean(this_contour[:,1]) ])
    return new_coord.astype(np.int)


@jit
def combine_markers(label_img, coordinates):
    #first we remove all the label_img region contain coordinates
    num_obj = np.max(label_img)

    regions = regionprops(label_img)

    seedmap = np.zeros_like(label_img, dtype=bool)
    seedmap[coordinates[:,0], coordinates[:,1]] = True

    max_num = num_obj + coordinates.shape[0]
    new_coord = np.zeros((max_num,2))
    seedcount = 0
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        thismask = label_img[minr:maxr, minc:maxc] == props.label
        this_seedmap  = seedmap[minr:maxr, minc:maxc]
        this_seeds = np.argwhere(np.logical_and(thismask, this_seedmap)) + np.array([[minr, minc]])

        number_seeds = this_seeds.shape[0]

        if number_seeds <= 1:
            new_coord[seedcount,:] = props.centroid
            seedcount += 1
        elif number_seeds >=2:
            new_coord[seedcount:seedcount+rem_cord.shape[0],:] = this_seeds
            seedcount += rem_cord.shape[0]
    return new_coord[0:seedcount,:].astype(np.int)

@jit
def intersect(arr_, brr_):
    #return the intersection of arr and brr.
    arr = set(map(tuple, arr_))
    brr = set(map(tuple, brr_))
    return np.asarray(arr.intersection(brr)).astype(np.int)


def residual_markers(label_img, coordinates):
    #first we remove all the label_img region contain coordinates
    # also need to return single label_imgage, and residual with markers
    num_obj = np.max(label_img)
    regions = regionprops(label_img)

    seedmap = np.zeros_like(label_img, dtype=bool)
    new_label = np.zeros_like(label_img)
    class_label = np.zeros_like(label_img)
    seedmap[coordinates[:,0], coordinates[:,1]] = True
    max_num = num_obj + coordinates.shape[0]
    #coordinates = set(map(tuple, coordinates))

    new_coord = np.zeros((max_num,2))
    seedcount = 0
    regionCount = 0
    classRegionCount = 0
    all_area = [props.area for props in regions]
    mid_area = np.median(all_area)
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        rlen = maxr-minr+1
        clen = maxc-minc+1
        ratio = props.minor_axis_length/props.major_axis_length

        thismask = label_img[minr:maxr, minc:maxc] == props.label
        this_seedmap  = seedmap[minr:maxr, minc:maxc]
        this_new_label  = new_label[minr:maxr, minc:maxc]
        this_class_label = class_label[minr:maxr, minc:maxc]
        this_seeds = np.argwhere(np.logical_and(thismask, this_seedmap)) + np.array([[minr, minc]])
        number_seeds = this_seeds.shape[0]

        if number_seeds <= 1:
            classRegionCount += 1
            this_class_label[thismask] = classRegionCount
            #coordinates.difference(map(tuple, this_seeds.tolist()))
        elif number_seeds >=2:
            # if the cell is very round, we don'r split it
            if ratio > 0.85 and props.solidity>0.9 and props.area < 4*mid_area:
                classRegionCount += 1
                this_class_label[thismask] = classRegionCount
            else:
                regionCount += 1
                this_new_label[thismask] = regionCount
                #rem_cord = intersect(props.coords, coordinates)
                new_coord[seedcount:seedcount+number_seeds,:] = this_seeds
                seedcount += number_seeds

    return class_label, new_label, new_coord[0:seedcount,:].astype(np.int)


@autojit
def safe_boarder(boarder_seed, row, col):
    '''
    board_seed: N*2 represent row and col for 0 and 1 axis.
    '''
    boarder_seed[boarder_seed[:,0] < 0, 0] = 0
    boarder_seed[boarder_seed[:,0] >= row,0]   = row-1
    boarder_seed[boarder_seed[:,1] < 0, 1]  = 0
    boarder_seed[boarder_seed[:,1] >= col, 1]  = col-1
    return boarder_seed

@autojit
def label2contour(label_img, org=None, print_color = [0,0,1], linewidth = 2, alpha = 1, returnImg = False):
    #npad = ((1,1),(1,1))
    row, col = label_img.shape
    #label_img = np.pad(label_img, npad, mode='constant', constant_values=0)
    contour_img = np.zeros(label_img.shape, dtype=bool)
    #tmp_img = np.zeros_like(label_img)
    regions = regionprops(label_img)
    contourlist = [np.array([-1,-1])]*len(regions) #because numba can not work with []
    for id, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        rs, re = max(minr-1,0), min(maxr+1, row)
        cs, ce = max(minc-1,0), min(maxc+1, col)
        thispatch = label_img[rs:re, cs:ce] == props.label
        contours  = measure.find_contours(thispatch, 0)
        thiscontour = (contours[0] + [rs, cs]).astype(int)

        contourlist[id] = safe_boarder(thiscontour, row, col)
        contour_img[thiscontour[:, 0], thiscontour[:, 1]] = True
    masked_img = None
    if returnImg:
        se = np.array([[ True,  True,  True],
                       [ True,  True,  True],
                       [ True,  True,  True]], dtype=bool)
        contour_mask = skimage.morphology.binary_dilation(contour_img, se)
        masked_img = overlayImg(org, contour_mask , print_color = print_color, alpha = alpha)
    return masked_img, contourlist

def split_img(img, windowsize=1000, board = 0, fixed_window = False, step_size = None, tuple_slice = False):
    '''
    img dimension: channel, row, col
    output:
        (IndexDict, PackList)
        IndexDict is a dictionry, the key is the actual patch size, the values is the list of identifier,
        PackList: list of (thisPatch,org_slice ,extract_slice, thisSize,identifier), the index of Packlist
        corresponds to the identifier.
        org_slice: slice coordinated at the orignal image.
        extract_slice: slice coordinate at the extract thisPatch,
        the length of org_slice should be equal to extract_slice.

        fixed_window: if true, it forces the extracted patches has to be of size window_size.
                      we don't pad the original image to make mod(imgsize, windowsize)==0, instead, if the remaining is small,
                      we expand the left board to lefter to compensate the smaller reminging patches.

                      The default behavior is False: get all window_size patches, and collect the remining patches as it is.

        step_size: if step_size is smaller than (windowsize-2*board), we extract the patches with overlapping.
                which means the org_slice is overlapping.

    eg:
    lenght = 17
    img = np.arange(2*lenght*lenght).reshape(2,lenght,lenght)

    nm = np.zeros(img.shape).astype(np.int)

    AllDict, PackList =  split_img(img, windowsize=7, board = 0, step_size= 2,fixed_window = True)

    print img

    print '---------------------------------------'

    print AllDict.keys()

    for key in AllDict.keys():
        iden_list = AllDict[key]
        for iden in iden_list:
            thispatch = PackList[iden][0]
            org_slice = PackList[iden][1]
            extract_slice = PackList[iden][2]

            nm[:,org_slice[0],org_slice[1]] = thispatch[:,extract_slice[0],extract_slice[1]]
            print thispatch[:,extract_slice[0],extract_slice[1]]
    print nm
    print sum(nm-img)
    '''
    IndexDict = {}
    identifier = -1
    PackList = []
    row_size, col_size = img.shape[1], img.shape[2]
    if windowsize is not None and  type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    if windowsize is None or (row_size <= windowsize[0] and col_size<=windowsize[1] and (not fixed_window)):
        pad_img = img
        rowsize, colsize = pad_img.shape[1:]

        org_slice = (slice(0, rowsize), slice(0, colsize))
        extract_slice = org_slice
        crop_patch_slice = (slice(0, rowsize), slice(0, colsize))
        thisSize =  (rowsize, colsize )
        identifier = identifier + 1

        org_slice_tuple = (0, 0)
        if thisSize in IndexDict:
           IndexDict[thisSize].append(identifier)
        else:
           IndexDict[thisSize] = []
           IndexDict[thisSize].append(identifier)
        PackList.append((crop_patch_slice, org_slice ,extract_slice, thisSize,identifier, org_slice_tuple))

    else:

        hidden_windowsize = (windowsize[0]-2*board, windowsize[1]-2*board)
        for each_size in hidden_windowsize:
            if each_size <= 0:
                raise RuntimeError('windowsize can not be smaller than board*2.')

        if type(step_size) is int:
            step_size = (step_size, step_size)
        if step_size is None:
            step_size = hidden_windowsize

        numRowblocks = int(math.ceil(float(row_size)/step_size[0]))
        numColblocks = int(math.ceil(float(col_size)/step_size[1]))

        # sanity check, make sure the image is at least of size window_size to the left-hand side if fixed_windows is true
        # which means,    -----*******|-----, left to the vertical board of original image is at least window_size.
        row_addition_board, col_addition_board = 0, 0
        addition_board = 0
        if fixed_window:
            if row_size + 2 * board < windowsize[0]: # means we need to add more on board.
                row_addition_board = windowsize[0] - (row_size + 2 * board )
            if col_size + 2 * board < windowsize[1]: # means we need to add more on board.
                col_addition_board = windowsize[1] - (col_size + 2 * board)
            addition_board = row_addition_board if row_addition_board > col_addition_board else col_addition_board

        left_pad = addition_board + board
        pad4d = ((0,0),( left_pad , board), ( left_pad , board ))
        pad_img = np.pad(img, pad4d, 'symmetric').astype(img.dtype)

        thisrowstart, thiscolstart =0, 0
        thisrowend, thiscolend = 0,0
        for row_idx in range(numRowblocks):
            thisrowlen = min(hidden_windowsize[0], row_size - row_idx * step_size[0])
            row_step_len = min(step_size[0], row_size - row_idx * step_size[0])

            thisrowstart = 0 if row_idx == 0 else thisrowstart + step_size[0]

            thisrowend = thisrowstart + thisrowlen

            row_shift = 0
            if fixed_window:
                if thisrowlen < hidden_windowsize[0]:
                    row_shift = hidden_windowsize[0] - thisrowlen

            for col_idx in range(numColblocks):
                thiscollen = min(hidden_windowsize[1], col_size -  col_idx * step_size[1])
                col_step_len = min(step_size[1], col_size - col_idx * step_size[1])

                thiscolstart = 0 if col_idx == 0 else thiscolstart + step_size[1]

                thiscolend = thiscolstart + thiscollen

                col_shift = 0
                if fixed_window:
                    # we need to shift the patch to left to make it at least windowsize.
                    if thiscollen < hidden_windowsize[1]:
                        col_shift = hidden_windowsize[1] - thiscollen

                #
                #----board----******************----board----
                #
                crop_r_start = thisrowstart - board - row_shift + left_pad
                crop_c_start = thiscolstart - board - col_shift + left_pad
                crop_r_end  =  thisrowend + board + left_pad
                crop_c_end  =  thiscolend + board + left_pad

                #we need to handle the tricky board condition
                # thispatch will be of size (:,:, windowsize+ 2*board)
                #thisPatch =  pad_img[:,crop_r_start:crop_r_end, crop_c_start:crop_c_end].copy()
                crop_patch_slice = (slice(crop_r_start, crop_r_end), slice(crop_c_start, crop_c_end))
                org_slice_tuple  = (crop_r_start-left_pad,  crop_c_start -left_pad )

                thisSize = (thisrowlen + 2*board + row_shift, thiscollen + 2*board + col_shift)


                org_slice = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))
                # slice on a cooridinate of the original image
                extract_slice = (slice(board + row_shift, board + thisrowlen + row_shift),
                                slice(board + col_shift, board + col_shift + thiscollen))
                # extract on local coordinate of a patch

                identifier =  identifier +1
                PackList.append((crop_patch_slice, org_slice ,extract_slice, thisSize,identifier, org_slice_tuple))

                if thisSize in IndexDict:
                   IndexDict[thisSize].append(identifier)
                else:
                   IndexDict[thisSize] = []
                   IndexDict[thisSize].append(identifier)

    PackDict = {}
    for this_size in list(IndexDict.keys()):
        iden_list = IndexDict[this_size]
        this_len = len(iden_list)
        org_slice_list = []
        extract_slice_list = []
        slice_tuple_list  = []
        BatchData = np.zeros( (this_len, img.shape[0]) + tuple(this_size) )
        for idx, iden in enumerate(iden_list):
            crop_patch_slice = PackList[iden][0]
            BatchData[idx,...] = pad_img[:,crop_patch_slice[0],crop_patch_slice[1]]
            org_slice_list.append(PackList[iden][1])
            extract_slice_list.append(PackList[iden][2])
            slice_tuple_list.append(PackList[iden][-1])

        PackDict[this_size]= (BatchData, org_slice_list, extract_slice_list, slice_tuple_list)

    return PackDict

def split_index(img, windowsize=1000, board = 0, fixed_window = False, step_size = None, tuple_slice = False):
    '''
    img dimension: channel, row, col
    output:
        (IndexDict, PackList)
        IndexDict is a dictionry, the key is the actual patch size, the values is the list of identifier,
        PackList: list of (thisPatch,org_slice ,extract_slice, thisSize,identifier), the index of Packlist
        corresponds to the identifier.
        org_slice: slice coordinated at the orignal image.
        extract_slice: slice coordinate at the extract thisPatch,
        the length of org_slice should be equal to extract_slice.

        fixed_window: if true, it forces the extracted patches has to be of size window_size.
                      we don't pad the original image to make mod(imgsize, windowsize)==0, instead, if the remaining is small,
                      we expand the left board to lefter to compensate the smaller reminging patches.

                      The default behavior is False: get all window_size patches, and collect the remining patches as it is.

        step_size: if step_size is smaller than (windowsize-2*board), we extract the patches with overlapping.
                which means the org_slice is overlapping.

    eg:
    lenght = 17
    img = np.arange(2*lenght*lenght).reshape(2,lenght,lenght)

    nm = np.zeros(img.shape).astype(np.int)

    AllDict, PackList =  split_img(img, windowsize=7, board = 0, step_size= 2,fixed_window = True)

    print img

    print '---------------------------------------'

    print AllDict.keys()

    for key in AllDict.keys():
        iden_list = AllDict[key]
        for iden in iden_list:
            thispatch = PackList[iden][0]
            org_slice = PackList[iden][1]
            extract_slice = PackList[iden][2]

            nm[:,org_slice[0],org_slice[1]] = thispatch[:,extract_slice[0],extract_slice[1]]
            print thispatch[:,extract_slice[0],extract_slice[1]]
    print nm
    print sum(nm-img)
    '''
    IndexDict = {}
    identifier = -1
    PackList = []
    row_size, col_size = img.shape[1], img.shape[2]
    if windowsize is not None and  type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    if windowsize is None or (row_size <= windowsize[0] and col_size<=windowsize[1] and (not fixed_window)):
        pad_img = img
        rowsize, colsize = pad_img.shape[1:]

        org_slice = (slice(0, rowsize), slice(0, colsize))
        extract_slice = org_slice
        crop_patch_slice = (slice(0, rowsize), slice(0, colsize))
        thisSize =  (rowsize, colsize )
        identifier = identifier + 1

        org_slice_tuple = (0, 0)
        if thisSize in IndexDict:
           IndexDict[thisSize].append(identifier)
        else:
           IndexDict[thisSize] = []
           IndexDict[thisSize].append(identifier)
        PackList.append((crop_patch_slice, org_slice ,extract_slice, thisSize,identifier, org_slice_tuple))

    else:

        hidden_windowsize = (windowsize[0]-2*board, windowsize[1]-2*board)
        for each_size in hidden_windowsize:
            if each_size <= 0:
                raise RuntimeError('windowsize can not be smaller than board*2.')

        if type(step_size) is int:
            step_size = (step_size, step_size)
        if step_size is None:
            step_size = hidden_windowsize

        numRowblocks = int(math.ceil(float(row_size)/step_size[0]))
        numColblocks = int(math.ceil(float(col_size)/step_size[1]))

        # sanity check, make sure the image is at least of size window_size to the left-hand side if fixed_windows is true
        # which means,    -----*******|-----, left to the vertical board of original image is at least window_size.
        row_addition_board, col_addition_board = 0, 0
        addition_board = 0
        if fixed_window:
            if row_size + 2 * board < windowsize[0]: # means we need to add more on board.
                row_addition_board = windowsize[0] - (row_size + 2 * board )
            if col_size + 2 * board < windowsize[1]: # means we need to add more on board.
                col_addition_board = windowsize[1] - (col_size + 2 * board)
            addition_board = row_addition_board if row_addition_board > col_addition_board else col_addition_board

        left_pad = addition_board + board
        pad4d = ((0,0),( left_pad , board), ( left_pad , board ))
        # forget about the 0 padding now.
        pad_img = np.pad(img, pad4d, 'symmetric').astype(img.dtype)
        
        thisrowstart, thiscolstart =0, 0
        thisrowend, thiscolend = 0,0
        for row_idx in range(numRowblocks):
            thisrowlen = min(hidden_windowsize[0], row_size - row_idx * step_size[0])
            row_step_len = min(step_size[0], row_size - row_idx * step_size[0])

            thisrowstart = 0 if row_idx == 0 else thisrowstart + step_size[0]

            thisrowend = thisrowstart + thisrowlen

            row_shift = 0
            if fixed_window:
                if thisrowlen < hidden_windowsize[0]:
                    row_shift = hidden_windowsize[0] - thisrowlen

            for col_idx in range(numColblocks):
                thiscollen = min(hidden_windowsize[1], col_size -  col_idx * step_size[1])
                col_step_len = min(step_size[1], col_size - col_idx * step_size[1])

                thiscolstart = 0 if col_idx == 0 else thiscolstart + step_size[1]

                thiscolend = thiscolstart + thiscollen

                col_shift = 0
                if fixed_window:
                    # we need to shift the patch to left to make it at least windowsize.
                    if thiscollen < hidden_windowsize[1]:
                        col_shift = hidden_windowsize[1] - thiscollen

                #
                #----board----******************----board----
                #
                crop_r_start = thisrowstart - board - row_shift + left_pad
                crop_c_start = thiscolstart - board - col_shift + left_pad
                crop_r_end  =  thisrowend + board + left_pad
                crop_c_end  =  thiscolend + board + left_pad

                #we need to handle the tricky board condition
                # thispatch will be of size (:,:, windowsize+ 2*board)
                #thisPatch =  pad_img[:,crop_r_start:crop_r_end, crop_c_start:crop_c_end].copy()
                crop_patch_slice = (slice(crop_r_start, crop_r_end), slice(crop_c_start, crop_c_end))
                org_slice_tuple  = (crop_r_start-left_pad,  crop_c_start -left_pad )

                thisSize = (thisrowlen + 2*board + row_shift, thiscollen + 2*board + col_shift)

                # slice on a cooridinate of the original image
                org_slice = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))

                # extract on local coordinate of a patch
                extract_slice = (slice(board + row_shift, board + thisrowlen + row_shift),
                                 slice(board + col_shift, board + col_shift + thiscollen))
                
                identifier =  identifier +1
                PackList.append((crop_patch_slice, org_slice, extract_slice, thisSize, identifier, org_slice_tuple))

                if thisSize in IndexDict:
                   IndexDict[thisSize].append(identifier)
                else:
                   IndexDict[thisSize] = []
                   IndexDict[thisSize].append(identifier)

    return  PackList, pad_img