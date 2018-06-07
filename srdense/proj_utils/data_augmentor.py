'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings

import numpy as np
from numpy.random import random_integers
from scipy.signal import convolve2d
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import sys

try:
   from .local_utils import imshow, imread
except:
   pass
from numba import jit 

def image_dim_ordering_():
    return 'th'
    
def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma
    
    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int
    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float
    
    :returns: a numpy 2d array
    """
    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = np.zeros((dim, dim), dtype=np.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2
    
    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)
    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance
            
            kernel[x,y] = coeff * np.exp(-1. * numerator/denom)
    # normalise it
    return kernel/sum(sum(kernel))


def elastic_transform(image_list, alpha=35, sigma=5, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    list_type = True
    if type(image_list) is not list:
        image_list = [image_list]
        list_type = False
    results = []
    
    shape = image_list[0].shape[-2::]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    for list_ind, this_org in enumerate(image_list):
        this_img = np.zeros_like(this_org)
        org_shape = this_org.shape
 
        if len(this_img.shape) == 2:
            this_img = this_img[None]
            this_org = this_org[None]
            
        for ind in range(this_img.shape[0]):
            this_img[ind] = map_coordinates(this_org[ind], indices, order=1).reshape(shape)
        this_img = this_img.reshape(org_shape)
        results.append(this_img)
    if not list_type:
        results = results[0]
    return results


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering_()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering_()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.

    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]

class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering_` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 elastic = False,
                 number_repeat=1, 
                 elastic_label = True,
                 transform_label = True,
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = image_dim_ordering_()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" (channel after row and '
                             'column) or "th" (channel before row and column). '
                             'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(self,
            X, y, 
            batch_size=batch_size, shuffle=shuffle, seed=seed, number_repeat = self.number_repeat,
            dim_ordering=self.dim_ordering, elastic_label= self.elastic_label,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, y=None, z = None):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        
        if y is not None:
            y_min, y_max = np.min(y), np.max(y)
        if z is not None:
            z_min, z_max = np.min(z), np.max(z)

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.transform_label:
            if y is not None:
                y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval) 
            if z is not None:
                z = apply_transform(z, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval) 
      

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                if self.transform_label:
                    if y is not None:
                        y = flip_axis(y, img_col_index)
                    if z is not None:
                        z = flip_axis(z, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                if self.transform_label:
                    if y is not None:
                        y = flip_axis(y, img_row_index) 
                    if z is not None:
                        z = flip_axis(z, img_row_index) 
        if self.elastic:
            
            if np.random.random() < 0.5:
                if self.transform_label and self.elastic_label:
                    if y is None and z is None:
                        x = elastic_transform(x)
                    elif y is not None and z is None:
                        x, y = elastic_transform([x,y])
                    elif y is None and z is not None:
                        x, z = elastic_transform([x,z])
                    else:
                        x = elastic_transform(x)
                else:
                    x = elastic_transform(x)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        if y is not None and z is not None:
            y[y>y_max] = y_max
            y[y<y_min] = y_min

            z[z>z_max] = z_max
            z[z<z_min] = z_min
            return x, y, z
        elif y is None and z is not None:
            z[z>z_max] = z_max
            z[z<z_min] = z_min
            return x, z
        elif y is not None and z is None:
            y[y>y_max] = y_max
            y[y<y_min] = y_min
            return x, y
        else:
            return x

class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed, number_repeat=1):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        #self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.number_repeat = number_repeat
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)
        
    def reset(self):
        self.batch_index = 0
        
    def _flow_index(self, N, batch_size=32, shuffle=True, seed=None):
        # ensure self.batch_index is 0
        number_repeat = 0
        self.reset()

        chunkidx = 0
        numberofchunk = int(N + batch_size - 1)// int(batch_size)   # the ceil

        while 1:
            batch_end = 2 #means we need to drop current loop               
            if seed is not None:
                np.random.seed(seed )
            if chunkidx == 0:
                current_index = 0
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(index_array)

            current_batch_size = min(batch_size, N - chunkidx*batch_size)
            thisInd = index_array[current_index: current_index + current_batch_size]
            old_current_index = current_index

            if chunkidx == numberofchunk-1:
                chunkidx = 0
                number_repeat += 1
                batch_end = True # because we dont want stopiteration before processing this, use this as a marker.
                current_index = 0
            else:
                batch_end = False
                current_index += current_batch_size
                chunkidx += 1
            yield (thisInd, old_current_index, current_batch_size, number_repeat, batch_end)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
        
class NumpyArrayIterator(Iterator):
    # in this case, the Y should also be an image.
    def __init__(self,image_data_generator, X, label_list=None,
                 batch_size=32, shuffle=True, seed=None,
                 dim_ordering='default', elastic_label=True,
                 save_to_dir=None, save_prefix='', save_format='jpeg', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = image_dim_ordering_()
        self.X = np.asarray(X)
        
        if self.X.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.X.shape)
        channels_axis = 3 if dim_ordering == 'tf' else 1

        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix 
        self.save_format = save_format
        self.elastic_label = elastic_label
        self.batch_x = np.zeros(tuple([batch_size] + list(self.X.shape)[1:]), dtype = 'float32')
        if label_list is not None:
            if type(label_list) is not list:
                label_list = [label_list]
            y = label_list[0]
            z = label_list[1] if len(label_list) >= 2 else None
            if y is not None:
                self.y = np.asarray(y)
                self.batch_y = np.zeros(tuple([batch_size] + list(self.y.shape)[1:]), dtype = 'float32')
            else:
                self.y = None
            
            if z is not None:
                self.z = np.asarray(z)
                self.batch_z = np.zeros(tuple([batch_size] + list(self.z.shape)[1:]), dtype = 'float32')
            else:
                self.z = None
        else:
            self.y = None
            self.z = None
        if y is not None and len(X) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                        'should have the same length. '
                        'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
                        
        self.last_batch_end = False
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed, **kwargs)

    def __next__(self): 
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size, number_repeat, batch_end= next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        
        if number_repeat >= self.number_repeat:
            if self.last_batch_end: 
                raise StopIteration()
            if batch_end:
                self.last_batch_end = True
                
        for i, j in enumerate(index_array):
            x = self.X[j]
            if self.z is not None and self.y is not None:
                z = self.z[j]
                y = self.y[j]
                x, y, z = self.image_data_generator.random_transform(x.astype('float32'), y.astype('float32'), z.astype('float32'))
            if self.z is None and self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype('float32'), y.astype('float32'))
            
            x = self.image_data_generator.standardize(x)

            self.batch_x[i] = x
            self.batch_y[i] = y
            if self.z is not None:
                self.batch_z[i] = z

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.z is None:           
            return self.batch_x[0:current_batch_size].astype('float32'), self.batch_y[0:current_batch_size].astype('float32')
        else:
            return self.batch_x[0:current_batch_size].astype('float32'), self.batch_y[0:current_batch_size].astype('float32'), \
                   self.batch_z[0:current_batch_size].astype('float32')


