import torch.utils.data as data
import torch
import h5py
from torch.multiprocessing import Pool
from scipy import ndimage
import scipy, skimage
from .proj_utils.local_utils import *

# import gdal
# def imread_(img_path):
#     img=gdal.Open(img_path)
#     reg_img=img.ReadAsArray()
#     reg_img = np.transpose(reg_img, (1,2,0))
#     return reg_img

def resize_thread(hd_img_orig, this_down_ratio):
    hd_row, hd_col, chn = hd_img_orig.shape
    reg_row, reg_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)
    
    reg_img_down = imresize_shape(hd_img_orig, (reg_row, reg_col) )
    reg_img_orig = imresize_shape(reg_img_down, (hd_row, hd_col) )
    
    hd_img  = (hd_img_orig/255 - 0.5 )  / 0.5
    reg_img = (reg_img_orig/255 - 0.5 ) / 0.5
    
    return (hd_img, reg_img)

def rotate_thread(inputs):
    this_hd_img, this_reg_img = inputs
    if random.random() > 0.3:
        angle = np.random.choice( np.arange(0, 360, 0.5), size=1, replace=False)[0]

        this_hd_img  = scipy.ndimage.interpolation.rotate(this_hd_img, angle, mode='reflect')
        this_reg_img = scipy.ndimage.interpolation.rotate(this_reg_img, angle, mode='reflect')

        this_hd_img[this_hd_img>1] = 1
        this_hd_img[this_hd_img<-1] = -1
        this_reg_img[this_reg_img>1] = 1
        this_reg_img[this_reg_img<-1] = -1

    return (this_hd_img, this_reg_img)

class DatasetFromfolder():
    def __init__(self, hd_folder, reg_folder=None, batch_size=64, ext_list=['.bmp', '.tif'],
                 img_size = 128, down_ratio= [0.45, 0.5, 0.55] ): # 
        super(DatasetFromfolder, self).__init__()
        self.hd_imgs  = []
        self.reg_imgs = []
        self.batch_size = batch_size
        self.ratio_pool = np.arange(0.45, 0.65, 0.01)
        img_count = 0
        self.hd_filelist, self.hd_filenames = getfilelist(hd_folder, ext_list)
        self.count = 0
        self.reset_count = 20000
        self.img_size = img_size
        self.epoch_iteration = 2000

        for this_hd_path, this_hd_name in zip(self.hd_filelist, self.hd_filenames):
            #this_reg = os.path.join(reg_folder, this_hd_name)
            hd_img_orig = imread(this_hd_path)

            hd_row, hd_col, chn = hd_img_orig.shape
            hd_row_start, hd_row_end = int(0.15*hd_row), int(0.85*hd_row)
            hd_col_start, hd_col_end = int(0.15*hd_row), int(0.85*hd_row)
            hd_img_orig  = hd_img_orig[hd_row_start:hd_row_end, hd_col_start:hd_col_end, :]

            hd_img  = (hd_img_orig/127.5 - 1 ).astype(np.float16)
            self.hd_imgs.append( hd_img )
            img_count +=  1
            print('Generated {} images'.format(img_count))

        self.num_imgs = len(self.hd_imgs)

    def get_next(self):
        # if self.count > 0 and self.count % self.reset_count == 0:
        #     self.reset_dataset()
        #     self.count = 0
        
        chose_img_index = np.random.choice(self.num_imgs, self.batch_size)
        chose_imgs = len(chose_img_index)
        total_data_reg = []
        total_data_hd  = []
        already_taken = 0
        every_taken = int(self.batch_size/chose_imgs)

        #import pdb; pdb.set_trace()
        for img_idx in chose_img_index:
            
            this_hd  = self.hd_imgs[img_idx]
            #this_reg = self.reg_imgs[img_idx]
            row_size, col_size = this_hd.shape[0:2]
            br = self.img_size//2 + 1
            this_chose = min(self.batch_size - already_taken , every_taken)

            already_taken = already_taken + this_chose

            #for idx in range(this_chose):
            this_chose_count = 0
            blank_count = 0
            while this_chose_count < this_chose:
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)

                #print(ridx, cidx)
                hd_data  = this_hd[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 

                if 0 and np.sum(hd_data  <  ( (210/255 - 0.5 ) / 0.5)  ) < 50 and blank_count < 1:
                    print('all blank area')
                    blank_count +=  1
                    pass 
                else:    
                    #reg_data = this_reg[ridx:ridx+self.img_size, cidx:cidx+self.img_size,:]
                    #print(hd_data.shape)
                    this_down_ratio =  np.random.choice(self.ratio_pool, size=1, replace=False)[0]
                    hd_row, hd_col, chn = hd_data.shape
                    reg_row, reg_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)
                    
                    reg_img_down = imresize_shape(hd_data, (reg_row, reg_col) )
                    #print(np.mean(reg_img_down))
                    reg_data = imresize_shape(reg_img_down, (hd_row, hd_col) )
                    sigma = np.random.choice([0, 0.5, 1, 1.5, 2], size=1, replace=False)[0]
                    #sigma = np.random.choice( [0, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5, 1.7, 1.9], size=1, replace=False)[0]
                    reg_data = skimage.filters.gaussian(reg_data, sigma=sigma,  multichannel=True)

                    #reg_data = (reg_data/127.5 - 1 )
                    
                    total_data_reg.append(reg_data) # N*5*5*3

                    total_data_hd.append(hd_data)   # N * 3
                    this_chose_count += 1
                    blank_count = 0
        #import pdb; pdb.set_trace()
        data_reg_np = np.asarray(total_data_reg, dtype=np.float32) 
        hd_np       = np.asarray(total_data_hd,  dtype=np.float32)
            
        data_reg_np = np.transpose(data_reg_np, (0, 3, 1, 2))
        hd_np = np.transpose(hd_np, (0, 3, 1, 2) )

        self.count  += self.batch_size
        return data_reg_np, hd_np


class DatasetSmall():
    def __init__(self, hd_folder, reg_folder=None, batch_size=64, ext_list=['.bmp', '.tif', '.png'], 
                 img_size = 128): # 
        super(DatasetSmall, self).__init__()
        self.hd_imgs_org  = []
        self.reg_imgs_org = []
        self.batch_size = batch_size
        self.ratio_pool = np.arange(0.5, 0.75, 0.01)
        img_count = 0
        self.hd_filelist, self.hd_filenames = getfilelist(hd_folder, ext_list, with_ext=True)
        self.count = 0
        self.reset_count = 20000
        self.img_size = img_size
        self.epoch_iteration = 5000
        self.pool = Pool(6)


        for this_hd_path, this_hd_name in zip(self.hd_filelist, self.hd_filenames):
            
            this_reg_path = os.path.join(reg_folder, this_hd_name)
            reg_img_orig = imread(this_reg_path).astype(np.float32)
            hd_img_orig  = imread(this_hd_path).astype(np.float32)

            hd_row, hd_col, chn = hd_img_orig.shape
            #reg_row, reg_col, chn = reg_img_orig.shape
            # now pad the image to at least img_size * 1.4
            pad_hd_row, pad_hd_col = max(0, int(1.4*img_size)-hd_row ), max(0, int(1.4*img_size) - hd_col  )
            
            npad3 = ((pad_hd_row//2, pad_hd_row-pad_hd_row//2),(pad_hd_col//2, pad_hd_col-pad_hd_col//2),(0,0))
            
            hd_img_orig  = np.pad(hd_img_orig, npad3, 'symmetric')
            reg_img_orig = np.pad(reg_img_orig, npad3, 'symmetric')

            hd_img  = (hd_img_orig/127.5 - 1 )
            reg_img = (reg_img_orig/127.5 - 1 )
            
            self.hd_imgs_org.append( hd_img )
            self.reg_imgs_org.append(reg_img)
            img_count +=  1
            print('Generated {} images'.format(img_count), hd_img_orig.shape)
            
        self.num_imgs = len(self.hd_imgs_org)
        print('start rotate')
        self.hd_imgs  = []
        self.reg_imgs = []
        self.rotate_img()

    def rotate_img(self):
        self.hd_imgs  = []
        self.reg_imgs = []
        
        targets = self.pool.imap(rotate_thread,
                ( (self.hd_imgs_org[img_idx], self.reg_imgs_org[img_idx] ) for i in range(self.num_imgs)) )                

        for img_idx in range(self.num_imgs): 
            this_hd_img, this_reg_img = targets.__next__()

            self.hd_imgs.append(this_hd_img)
            self.reg_imgs.append(this_reg_img)

    def get_next(self):
        if self.count > 0 and self.count % self.reset_count == 0:
            self.rotate_img()
            self.count = 0
        
        chose_img_index = np.random.choice(self.num_imgs, self.batch_size)
        chose_imgs = len(chose_img_index)
        
        hd_data_list  = []
        reg_data_list = []

        already_taken = 0
        every_taken = int(self.batch_size/chose_imgs)

        for img_idx in chose_img_index:
            
            this_hd  = self.hd_imgs[img_idx]
            this_reg = self.reg_imgs[img_idx]
            # rotate the image

            br = self.img_size//2 + 1
            this_chose = min(self.batch_size - already_taken , every_taken)
            
            already_taken = already_taken + this_chose

            #for idx in range(this_chose):
            this_chose_count = 0
            blank_count = 0
            
            while this_chose_count < this_chose:
                
                row_size, col_size = this_hd.shape[0:2]
                try:
                    ridx = random.randint(0 , row_size - self.img_size - 1)
                    cidx = random.randint(0,  col_size -  self.img_size - 1)
                    
                    hd_data  = this_hd[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 

                    if random.random() > 0.3:
                        reg_data  = this_reg[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 
                    else: #sometime, I just use true data and make it small to train the model
                        this_down_ratio =  np.random.choice(self.ratio_pool, size=1, replace=False)[0]
                        hd_row, hd_col, chn = hd_data.shape
                        low_row, low_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)

                        low_hd = imresize_shape(hd_data, (low_row, low_col) )
                        #print(np.mean(reg_img_down))
                        low_hd = imresize_shape(low_hd, (hd_row, hd_col) )
                        #print('max and min value of hd image: ', np.max(low_hd), np.min(low_hd))
                        sigma = np.random.choice( [1,2], size=1, replace=False)[0]
                        reg_data = skimage.filters.gaussian(low_hd.astype(np.float32), sigma=sigma,  multichannel=True)
                         
                    hd_data_list.append(hd_data)
                    reg_data_list.append(reg_data)
                except:
                    print('we have problem cropping the training data, img_size: ',row_size, col_size )
                
                this_chose_count += 1
                blank_count = 0
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        hd_data_np       = np.asarray(hd_data_list, dtype=np.float32) 
        reg_data_np      = np.asarray(reg_data_list,  dtype=np.float32)

        hd_data_np = np.transpose(hd_data_np, (0, 3, 1, 2))
        reg_data_np = np.transpose(reg_data_np, (0, 3, 1, 2) )  
        self.count  += self.batch_size

        return  {'low': reg_data_np, 'high': hd_data_np} 


class DatasetGAN():
    def __init__(self, hd_folder, reg_folder=None, batch_size=64, ext_list=['.bmp', '.tif'], img_size = 128): # 
        super(DatasetGAN, self).__init__()
        self.hd_imgs  = []
        self.reg_imgs = []
        self.batch_size = batch_size
        self.ratio_pool = np.arange(0.5, 0.75, 0.01)
        img_count = 0
        self.hd_filelist, self.hd_filenames = getfilelist(hd_folder, ext_list, with_ext=True)
        self.count = 0
        self.reset_count = 20000
        self.img_size = img_size
        self.epoch_iteration = 2000

        for this_hd_path, this_hd_name in zip(self.hd_filelist, self.hd_filenames):
            
            this_reg_path = os.path.join(reg_folder, this_hd_name)
            reg_img_orig = imread(this_reg_path)
            hd_img_orig  = imread(this_hd_path)

            hd_row, hd_col, chn = hd_img_orig.shape
            reg_row, reg_col, chn = reg_img_orig.shape

            hd_row_start, hd_row_end = int(0.15*hd_row), int(0.85*hd_row)
            hd_col_start, hd_col_end = int(0.15*hd_row), int(0.85*hd_row)
            
            reg_row_start, reg_row_end = int(0.15*reg_row), int(0.85*reg_row)
            reg_col_start, reg_col_end = int(0.15*reg_row), int(0.85*reg_row)

            hd_img_orig  = hd_img_orig[hd_row_start:hd_row_end, hd_col_start:hd_col_end, :]
            reg_img_orig = reg_img_orig[reg_row_start:reg_row_end, reg_col_start:reg_col_end, :]

            hd_img  = (hd_img_orig/127.5 - 1 ).astype(np.float16)
            reg_img = (reg_img_orig/127.5 - 1 ).astype(np.float16)
            

            self.hd_imgs.append( hd_img )
            self.reg_imgs.append(reg_img)
            img_count +=  1
            print('Generated {} images'.format(img_count))
            
        self.num_imgs = len(self.hd_imgs)

    def get_next(self):
        # if self.count > 0 and self.count % self.reset_count == 0:
        #     self.reset_dataset()
        #     self.count = 0
        
        chose_img_index = np.random.choice(self.num_imgs, self.batch_size)
        chose_imgs = len(chose_img_index)
        total_low_hd = []
        total_data_hd  = []
        unpaired_reg = []

        already_taken = 0
        every_taken = int(self.batch_size/chose_imgs)

        #import pdb; pdb.set_trace()
        for img_idx in chose_img_index:
            
            this_hd  = self.hd_imgs[img_idx]
            this_reg = self.reg_imgs[img_idx]

            br = self.img_size//2 + 1
            this_chose = min(self.batch_size - already_taken , every_taken)
            
            already_taken = already_taken + this_chose

            #for idx in range(this_chose):
            this_chose_count = 0
            blank_count = 0
            while this_chose_count < this_chose:
                row_size, col_size = this_hd.shape[0:2]
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)
                hd_data  = this_hd[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 

                row_size, col_size = this_reg.shape[0:2]
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)
                reg_data  = this_reg[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 
                
                unpaired_reg.append(reg_data)

                #reg_data = this_reg[ridx:ridx+self.img_size, cidx:cidx+self.img_size,:]
                #print(hd_data.shape)
                this_down_ratio =  np.random.choice(self.ratio_pool, size=1, replace=False)[0]
                hd_row, hd_col, chn = hd_data.shape
                low_row, low_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)
                
                low_hd = imresize_shape(hd_data, (low_row, low_col) )
                #print(np.mean(reg_img_down))
                low_hd_data = imresize_shape(low_hd, (hd_row, hd_col) )
                sigma = np.random.choice( [0, 0.5, 1, 2], size=1, replace=False)[0]
                low_hd_data = skimage.filters.gaussian(low_hd_data, sigma=sigma,  multichannel=True)

                #reg_data = (reg_data/127.5 - 1 )
                total_low_hd.append(low_hd_data) # N*5*5*3

                total_data_hd.append(hd_data)   # N * 3
                this_chose_count += 1
                blank_count = 0
        #import pdb; pdb.set_trace()
        data_low_hd_np = np.asarray(total_low_hd, dtype=np.float32) 
        hd_np       = np.asarray(total_data_hd,  dtype=np.float32)
        unpaird_reg_np  = np.asarray(unpaired_reg,  dtype=np.float32)

        data_low_hd_np = np.transpose(data_low_hd_np, (0, 3, 1, 2))
        hd_np = np.transpose(hd_np, (0, 3, 1, 2) )
        unpaird_reg_np = np.transpose(unpaird_reg_np, (0, 3, 1, 2) )

        batch_dict = {'A': unpaird_reg_np, 'B': hd_np, 'low_B':data_low_hd_np}
        self.count  += self.batch_size
        return batch_dict


class DatasetPairGAN(): # unfinished yet.
    def __init__(self, hd_folder, reg_folder=None, batch_size=64, ext_list=['.bmp', '.tif'], img_size = 128): # 
        super(DatasetPairGAN, self).__init__()
        self.hd_imgs  = []
        self.reg_imgs = []
        self.batch_size = batch_size
        self.ratio_pool = np.arange(0.5, 0.75, 0.01)
        img_count = 0
        self.hd_filelist, self.hd_filenames = getfilelist(hd_folder, ext_list, with_ext=True)
        self.count = 0
        self.reset_count = 20000
        self.img_size = img_size
        self.epoch_iteration = 5000

        for this_hd_path, this_hd_name in zip(self.hd_filelist, self.hd_filenames):
            
            this_reg_path = os.path.join(reg_folder, this_hd_name)
            reg_img_orig = imread(this_reg_path)
            hd_img_orig  = imread(this_hd_path)

            hd_row, hd_col, chn = hd_img_orig.shape
            reg_row, reg_col, chn = reg_img_orig.shape

            hd_row_start, hd_row_end = int(0.15*hd_row), int(0.85*hd_row)
            hd_col_start, hd_col_end = int(0.15*hd_row), int(0.85*hd_row)
            
            reg_row_start, reg_row_end = int(0.15*reg_row), int(0.85*reg_row)
            reg_col_start, reg_col_end = int(0.15*reg_row), int(0.85*reg_row)

            hd_img_orig  = hd_img_orig[hd_row_start:hd_row_end, hd_col_start:hd_col_end, :]
            reg_img_orig = reg_img_orig[reg_row_start:reg_row_end, reg_col_start:reg_col_end, :]

            hd_img  = (hd_img_orig/127.5 - 1 ).astype(np.float16)
            reg_img = (reg_img_orig/127.5 - 1 ).astype(np.float16)
            

            self.hd_imgs.append( hd_img )
            self.reg_imgs.append(reg_img)
            img_count +=  1
            print('Generated {} images'.format(img_count))
            
        self.num_imgs = len(self.hd_imgs)

    def get_next(self):
        # if self.count > 0 and self.count % self.reset_count == 0:
        #     self.reset_dataset()
        #     self.count = 0
        
        chose_img_index = np.random.choice(self.num_imgs, self.batch_size)
        chose_imgs = len(chose_img_index)
        total_low_hd = []
        total_data_hd  = []
        unpaired_reg = []

        total_true_pair_hd = []  # form true pair
        already_taken = 0
        every_taken = int(self.batch_size/chose_imgs)

        #import pdb; pdb.set_trace()
        for img_idx in chose_img_index:
            
            this_hd  = self.hd_imgs[img_idx]
            this_reg = self.reg_imgs[img_idx]

            br = self.img_size//2 + 1
            this_chose = min(self.batch_size - already_taken , every_taken)
            
            already_taken = already_taken + this_chose

            #for idx in range(this_chose):
            this_chose_count = 0
            blank_count = 0
            while this_chose_count < this_chose:
                row_size, col_size = this_hd.shape[0:2]
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)
                hd_data  = this_hd[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 
                total_data_hd.append(hd_data)   # N * 3
                
                row_size, col_size = this_reg.shape[0:2]
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)
                reg_data  = this_reg[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 
                unpaired_reg.append(reg_data)

                # get another hd patch as true pair. 
                row_size, col_size = this_hd.shape[0:2]
                ridx = random.randint(0 + int(0.2*row_size), row_size - int(0.2*row_size)- self.img_size - 1)
                cidx = random.randint(0 + int(0.2*col_size), col_size - int(0.2*col_size) -  self.img_size - 1)
                true_hd_data  = this_hd[ridx:ridx+self.img_size,  cidx:cidx+self.img_size,:] 
                total_true_pair_hd.append(true_hd_data)   # N * 3

                #reg_data = this_reg[ridx:ridx+self.img_size, cidx:cidx+self.img_size,:]
                #print(hd_data.shape)
                this_down_ratio =  np.random.choice(self.ratio_pool, size=1, replace=False)[0]
                hd_row, hd_col, chn = hd_data.shape
                low_row, low_col = int(this_down_ratio*hd_row), int(this_down_ratio*hd_col)
                
                low_hd = imresize_shape(hd_data, (low_row, low_col) )
                #print(np.mean(reg_img_down))
                low_hd_data = imresize_shape(low_hd, (hd_row, hd_col) )
                sigma = np.random.choice( [1,2], size=1, replace=False)[0]
                low_hd_data = skimage.filters.gaussian(low_hd_data, sigma=sigma,  multichannel=True)

                #reg_data = (reg_data/127.5 - 1 )
                total_low_hd.append(low_hd_data) # N*5*5*3

                
                this_chose_count += 1
                blank_count = 0
        #import pdb; pdb.set_trace()
        data_low_hd_np = np.asarray(total_low_hd, dtype=np.float32) 
        hd_np       = np.asarray(total_data_hd,  dtype=np.float32)
        unpaird_reg_np  = np.asarray(unpaired_reg,  dtype=np.float32)

        data_low_hd_np = np.transpose(data_low_hd_np, (0, 3, 1, 2))
        hd_np = np.transpose(hd_np, (0, 3, 1, 2) )
        unpaird_reg_np = np.transpose(unpaird_reg_np, (0, 3, 1, 2) )
        
        batch_dict = {'A': unpaird_reg_np, 'B': hd_np, 'low_B':data_low_hd_np}
        self.count  += self.batch_size
        return batch_dict

