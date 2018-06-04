from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import os
import cv2
from keras.layers import Input,Dense,Flatten,Dropout,concatenate,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import nibabel as nib
import math
import tensorflow as tf


batch_size = 32
aaa=np.loadtxt('maxANDmin.txt')

def get_images():

    x,y = 173,173
    z=range(0,207)
    sliced_z = 207
    full_z = 207
    new_z = len(z)
    resizeTo=176

    ff = os.listdir("./ground3T")
    t = []
    ToPredict_images=[]
    Tgx_images=[]
    predict_matrix=[]
    Tgy_images=[]

    ground_images=[]
    Tgxx_images_g=[]
    Tgyx_images_g=[]
    Tgxx_images=[]
    Tgyx_images=[]
    Tgxx_images=np.asarray(Tgxx_images)
    Tgyx_images=np.asarray(Tgyx_images)
    Tgx_images_g=[]
    ground_matrix=[]
    Tgy_images_g=[]

    # UPLOAD THE TEST 3t AND TEST 7t IMAGES (GROUND)
    for f in ff:
    	temp = np.zeros([resizeTo,new_z,resizeTo])
    	a = nib.load("./ground3T/" + f)
    	affine = a.affine
    	a = a.get_data()
    	temp[3:,:,3:] = a
    	a = temp
    	Tgxx_images=[]
    	Tgyx_images=[]
    	for k in range(full_z):
    		Tgxx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
    		Tgyx_images.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
    	Tgxx_images=np.asarray(Tgxx_images)
    	Tgyx_images=np.asarray(Tgyx_images)
    	for j in range(full_z):
    		predict_matrix.append(a[:,j,:])
    		Tgx_images.append(Tgxx_images[j,:,:])
    		Tgy_images.append(Tgyx_images[j,:,:])

    predict_matrix = np.asarray(predict_matrix)
    Tgx_images=np.asarray(Tgx_images)
    Tgy_images=np.asarray(Tgy_images)
    ToPredict_images=np.zeros(shape=[(predict_matrix.shape[0]),(predict_matrix.shape[1]),(predict_matrix.shape[2]),(3)])
    for i in range(predict_matrix.shape[0]):
    	ToPredict_images[i,:,:,0] = predict_matrix[i,:,:].reshape(resizeTo,resizeTo)
    	ToPredict_images[i,:,:,1] = Tgx_images[i,:,:].reshape(resizeTo,resizeTo)
    	ToPredict_images[i,:,:,2] = Tgy_images[i,:,:].reshape(resizeTo,resizeTo)

    ToPredict_images = np.asarray(ToPredict_images)
    ToPredict_images = ToPredict_images.astype('float32')
    mx = np.max(ToPredict_images[:,:,:,0])
    mn = np.min(ToPredict_images[:,:,:,0])
    ToPredict_images[:,:,:,0] = (ToPredict_images[:,:,:,0] - mn ) / (mx - mn)
    mx = np.max(ToPredict_images[:,:,:,1])
    mn = np.min(ToPredict_images[:,:,:,1])
    ToPredict_images[:,:,:,1] = (ToPredict_images[:,:,:,1] - mn ) / (mx - mn)
    mx = np.max(ToPredict_images[:,:,:,2])
    mn = np.min(ToPredict_images[:,:,:,2])
    ToPredict_images[:,:,:,2] = (ToPredict_images[:,:,:,2] - mn ) / (mx - mn)





    ff = os.listdir("./ground7T")
    for f in ff:
    	temp = np.zeros([resizeTo,new_z,resizeTo])
    	a = nib.load("./ground7T/" + f)
    	affine = a.affine
    	a = a.get_data()
    	temp[3:,:,3:] = a
    	a = temp
    	Tgxx_images_g=[]
    	Tgyx_images_g=[]
    	for k in range(full_z):
    		Tgxx_images_g.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,1,0,ksize=5))
    		Tgyx_images_g.append(cv2.Sobel(np.reshape(a[:,k,:],[resizeTo,resizeTo]),cv2.CV_64F,0,1,ksize=5))
    	Tgxx_images_g=np.asarray(Tgxx_images_g)
    	Tgyx_images_g=np.asarray(Tgyx_images_g)
    	for j in range(full_z):
    		ground_matrix.append(a[:,j,:])
    		Tgx_images_g.append(Tgxx_images_g[j,:,:])
    		Tgy_images_g.append(Tgyx_images_g[j,:,:])

    ground_matrix=np.asarray(ground_matrix)
    Tgx_images_g=np.asarray(Tgx_images_g)
    Tgy_images_g=np.asarray(Tgy_images_g)
    ground_images=np.zeros(shape=[(ground_matrix.shape[0]),(ground_matrix.shape[1]),(ground_matrix.shape[2]),(3)])

    for i in range(ground_matrix.shape[0]):
    	ground_images[i,:,:,0] = ground_matrix[i,:,:].reshape(resizeTo,resizeTo)
    	ground_images[i,:,:,1] = Tgx_images_g[i,:,:].reshape(resizeTo,resizeTo)
    	ground_images[i,:,:,2] = Tgy_images_g[i,:,:].reshape(resizeTo,resizeTo)


    ground_images = np.asarray(ground_images)
    ground_images = ground_images.astype('float32')
    mx = np.max(ground_images[:,:,:,0])
    mn = np.min(ground_images[:,:,:,0])
    ground_images[:,:,:,0] = (ground_images[:,:,:,0] - mn ) / (mx - mn)
    mx = np.max(ground_images[:,:,:,1])
    mn = np.min(ground_images[:,:,:,1])
    ground_images[:,:,:,1] = (ground_images[:,:,:,1] - mn ) / (mx - mn)
    mx = np.max(ground_images[:,:,:,2])
    mn = np.min(ground_images[:,:,:,2])
    ground_images[:,:,:,2] = (ground_images[:,:,:,2] - mn ) / (mx - mn)


    cv2.imwrite("img2222.png",ToPredict_images[60,:,:,0]*255.0)
    cv2.imwrite("img2222_gt.png",ground_images[60,:,:,0]*255.0)
    return ToPredict_images, ground_images


epochs=1000
path1 = ''
learning_rate = 0.00001
lr = tf.Variable(0.1)
saver = tf.train.Saver()
epoch_loss1=0


p1 = tf.placeholder(shape=(None,176,176,3),dtype=tf.float32)
p2 = tf.placeholder(shape=(None,176,176,3),dtype=tf.float32)

def bootstrap_net(input_):

    '''Encoder'''

    conv1 = tf.layers.conv2d(inputs = input_, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv1")
    conv1_1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv1_1")
    conv1_2 = tf.layers.conv2d(inputs = conv1_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv1_2")
    conv1_3 = tf.layers.batch_normalization(inputs = conv1_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv1_3")
    conv1_4 = tf.layers.conv2d(inputs = conv1_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv1_4")
    conv1_5 = tf.layers.batch_normalization(inputs = conv1_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv1_5")
    pool1 = tf.layers.max_pooling2d(inputs = conv1_5, pool_size = 2, strides=2, name = "max_pool1")

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv2")
    conv2_1 = tf.layers.batch_normalization(inputs = conv2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv2_1")
    conv2_2 = tf.layers.conv2d(inputs = conv2_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv2_2")
    conv2_3 = tf.layers.batch_normalization(inputs = conv2_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv2_3")
    conv2_4 = tf.layers.conv2d(inputs = conv2_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv2_4")
    conv2_5 = tf.layers.batch_normalization(inputs = conv2_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv2_5")
    pool2 = tf.layers.max_pooling2d(inputs = conv2_5, pool_size = 2, strides=2, name = "max_pool2")

    conv3 = tf.layers.conv2d(inputs = pool2, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv3")
    conv3_1 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv3_1")
    conv3_2 = tf.layers.conv2d(inputs = conv3_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv3_2")
    conv3_3 = tf.layers.batch_normalization(inputs = conv3_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv3_3")
    conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv3_4")
    conv3_5 = tf.layers.batch_normalization(inputs = conv3_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv3_5")
    pool3 = tf.layers.max_pooling2d(inputs = conv3_5, pool_size = 2, strides=2, name = "max_pool3")

    conv4 = tf.layers.conv2d(inputs = pool3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv4")
    conv4_1 = tf.layers.batch_normalization(inputs = conv4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv4_1")
    conv4_2 = tf.layers.conv2d(inputs = conv4_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv4_2")
    conv4_3 = tf.layers.batch_normalization(inputs = conv4_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv4_3")
    conv4_4 = tf.layers.conv2d(inputs = conv4_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv4_4")
    conv4_5 = tf.layers.batch_normalization(inputs = conv4_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv4_5")

    conv5 = tf.layers.conv2d(inputs = conv4_5, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv5")
    conv5_1 = tf.layers.batch_normalization(inputs = conv5, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv5_1")
    conv5_2 = tf.layers.conv2d(inputs = conv5_1, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv5_2")
    conv5_3 = tf.layers.batch_normalization(inputs = conv5_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv5_3")
    conv5_4 = tf.layers.conv2d(inputs = conv5_3, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv5_4")
    conv5_5 = tf.layers.batch_normalization(inputs = conv5_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv5_5")

    '''Decoder'''
    up6 = tf.concat([conv5_5, conv4_5], axis = 3)
    conv6 = tf.layers.conv2d(inputs = up6, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv6")
    conv6_1 = tf.layers.batch_normalization(inputs = conv6, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv6_1")
    conv6_2 = tf.layers.conv2d(inputs = conv6_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv6_2")
    conv6_3 = tf.layers.batch_normalization(inputs = conv6_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv6_3")
    conv6_4 = tf.layers.conv2d(inputs = conv6_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv6_4")
    conv6_5 = tf.layers.batch_normalization(inputs = conv6_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv6_5")

    up7 = tf.keras.layers.UpSampling2D(2)(conv6_5)
    up7 = tf.concat([up7, conv3_5], axis = 3)
    conv7 = tf.layers.conv2d(inputs = up7, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv7")
    conv7_1 = tf.layers.batch_normalization(inputs = conv7, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv7_1")
    conv7_2 = tf.layers.conv2d(inputs = conv7_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv7_2")
    conv7_3 = tf.layers.batch_normalization(inputs = conv7_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv7_3")
    conv7_4 = tf.layers.conv2d(inputs = conv7_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv7_4")
    conv7_5 = tf.layers.batch_normalization(inputs = conv7_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv7_5")

    up8 = tf.keras.layers.UpSampling2D(2)(conv7_5)
    up8 = tf.concat([up8, conv2_5], axis = 3)
    conv8 = tf.layers.conv2d(inputs = up8, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv8")
    conv8_1 = tf.layers.batch_normalization(inputs = conv8, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv8_1")
    conv8_2 = tf.layers.conv2d(inputs = conv8_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv8_2")
    conv8_3 = tf.layers.batch_normalization(inputs = conv8_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv8_3")
    conv8_4 = tf.layers.conv2d(inputs = conv8_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv8_4")
    conv8_5 = tf.layers.batch_normalization(inputs = conv8_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv8_5")

    up9 = tf.keras.layers.UpSampling2D(2)(conv8_5)
    up9 = tf.concat([up9, conv1_5], axis = 3)
    conv9 = tf.layers.conv2d(inputs = up9, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv9")
    conv9_1 = tf.layers.batch_normalization(inputs = conv9, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv9_1")
    conv9_2 = tf.layers.conv2d(inputs = conv9_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv9_2")
    conv9_3 = tf.layers.batch_normalization(inputs = conv9_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv9_3")
    conv9_4 = tf.layers.conv2d(inputs = conv9_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "boot_conv9_4")
    conv9_5 = tf.layers.batch_normalization(inputs = conv9_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "boot_conv9_5")

    decoded =  tf.layers.conv2d(inputs = conv9_5, filters = 3, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.sigmoid, name = "decoded_boot")

    return decoded

def iterative_net(input_, reuse=False):

    '''Encoder'''

    conv1 = tf.layers.conv2d(inputs = input_, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv1", reuse=reuse)
    conv1_1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv1_1", reuse = reuse)
    conv1_2 = tf.layers.conv2d(inputs = conv1_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv1_2", reuse=reuse)
    conv1_3 = tf.layers.batch_normalization(inputs = conv1_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv1_3", reuse = reuse)
    conv1_4 = tf.layers.conv2d(inputs = conv1_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv1_4", reuse=reuse)
    conv1_5 = tf.layers.batch_normalization(inputs = conv1_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv1_5", reuse = reuse)
    pool1 = tf.layers.max_pooling2d(inputs = conv1_5, pool_size = 2, strides=2, name = "max_pool1")

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv2", reuse=reuse)
    conv2_1 = tf.layers.batch_normalization(inputs = conv2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv2_1", reuse = reuse)
    conv2_2 = tf.layers.conv2d(inputs = conv2_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv2_2", reuse=reuse)
    conv2_3 = tf.layers.batch_normalization(inputs = conv2_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv2_3", reuse = reuse)
    conv2_4 = tf.layers.conv2d(inputs = conv2_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv2_4", reuse=reuse)
    conv2_5 = tf.layers.batch_normalization(inputs = conv2_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv2_5", reuse = reuse)
    pool2 = tf.layers.max_pooling2d(inputs = conv2_5, pool_size = 2, strides=2, name = "max_pool2")

    conv3 = tf.layers.conv2d(inputs = pool2, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv3", reuse=reuse)
    conv3_1 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv3_1", reuse = reuse)
    conv3_2 = tf.layers.conv2d(inputs = conv3_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv3_2", reuse=reuse)
    conv3_3 = tf.layers.batch_normalization(inputs = conv3_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv3_3", reuse = reuse)
    conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv3_4", reuse=reuse)
    conv3_5 = tf.layers.batch_normalization(inputs = conv3_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv3_5", reuse = reuse)
    pool3 = tf.layers.max_pooling2d(inputs = conv3_5, pool_size = 2, strides=2, name = "max_pool3")

    conv4 = tf.layers.conv2d(inputs = pool3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv4", reuse=reuse)
    conv4_1 = tf.layers.batch_normalization(inputs = conv4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv4_1", reuse = reuse)
    conv4_2 = tf.layers.conv2d(inputs = conv4_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv4_2", reuse=reuse)
    conv4_3 = tf.layers.batch_normalization(inputs = conv4_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv4_3", reuse = reuse)
    conv4_4 = tf.layers.conv2d(inputs = conv4_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv4_4", reuse=reuse)
    conv4_5 = tf.layers.batch_normalization(inputs = conv4_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv4_5", reuse = reuse)

    conv5 = tf.layers.conv2d(inputs = conv4_5, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv5", reuse=reuse)
    conv5_1 = tf.layers.batch_normalization(inputs = conv5, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv5_1", reuse = reuse)
    conv5_2 = tf.layers.conv2d(inputs = conv5_1, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv5_2", reuse=reuse)
    conv5_3 = tf.layers.batch_normalization(inputs = conv5_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv5_3", reuse = reuse)
    conv5_4 = tf.layers.conv2d(inputs = conv5_3, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv5_4", reuse=reuse)
    conv5_5 = tf.layers.batch_normalization(inputs = conv5_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv5_5", reuse = reuse)

    '''Decoder'''
    up6 = tf.concat([conv5_5, conv4_5], axis = 3)
    conv6 = tf.layers.conv2d(inputs = up6, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv6", reuse=reuse)
    conv6_1 = tf.layers.batch_normalization(inputs = conv6, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv6_1", reuse = reuse)
    conv6_2 = tf.layers.conv2d(inputs = conv6_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv6_2", reuse=reuse)
    conv6_3 = tf.layers.batch_normalization(inputs = conv6_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv6_3", reuse = reuse)
    conv6_4 = tf.layers.conv2d(inputs = conv6_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv6_4", reuse=reuse)
    conv6_5 = tf.layers.batch_normalization(inputs = conv6_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv6_5", reuse = reuse)

    up7 = tf.keras.layers.UpSampling2D(2)(conv6_5)
    up7 = tf.concat([up7, conv3_5], axis = 3)
    conv7 = tf.layers.conv2d(inputs = up7, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv7", reuse=reuse)
    conv7_1 = tf.layers.batch_normalization(inputs = conv7, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv7_1", reuse = reuse)
    conv7_2 = tf.layers.conv2d(inputs = conv7_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv7_2", reuse=reuse)
    conv7_3 = tf.layers.batch_normalization(inputs = conv7_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv7_3", reuse = reuse)
    conv7_4 = tf.layers.conv2d(inputs = conv7_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv7_4", reuse=reuse)
    conv7_5 = tf.layers.batch_normalization(inputs = conv7_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv7_5", reuse = reuse)

    up8 = tf.keras.layers.UpSampling2D(2)(conv7_5)
    up8 = tf.concat([up8, conv2_5], axis = 3)
    conv8 = tf.layers.conv2d(inputs = up8, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv8", reuse=reuse)
    conv8_1 = tf.layers.batch_normalization(inputs = conv8, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv8_1", reuse = reuse)
    conv8_2 = tf.layers.conv2d(inputs = conv8_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv8_2", reuse=reuse)
    conv8_3 = tf.layers.batch_normalization(inputs = conv8_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv8_3", reuse = reuse)
    conv8_4 = tf.layers.conv2d(inputs = conv8_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv8_4", reuse=reuse)
    conv8_5 = tf.layers.batch_normalization(inputs = conv8_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv8_5", reuse = reuse)

    up9 = tf.keras.layers.UpSampling2D(2)(conv8_5)
    up9 = tf.concat([up9, conv1_5], axis = 3)
    conv9 = tf.layers.conv2d(inputs = up9, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv9", reuse=reuse)
    conv9_1 = tf.layers.batch_normalization(inputs = conv9, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv9_1", reuse = reuse)
    conv9_2 = tf.layers.conv2d(inputs = conv9_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv9_2", reuse=reuse)
    conv9_3 = tf.layers.batch_normalization(inputs = conv9_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv9_3", reuse = reuse)
    conv9_4 = tf.layers.conv2d(inputs = conv9_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "iter_conv9_4", reuse=reuse)
    conv9_5 = tf.layers.batch_normalization(inputs = conv9_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "iter_conv9_5", reuse = reuse)

    decoded =  tf.layers.conv2d(inputs = conv9_5, filters = 3, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.sigmoid, name = "decoded_iter", reuse=reuse)

    return decoded

def refine_net(input_):

    '''Encoder'''

    conv1 = tf.layers.conv2d(inputs = input_, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv1")
    conv1_1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv1_1")
    conv1_2 = tf.layers.conv2d(inputs = conv1_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv1_2")
    conv1_3 = tf.layers.batch_normalization(inputs = conv1_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv1_3")
    conv1_4 = tf.layers.conv2d(inputs = conv1_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv1_4")
    conv1_5 = tf.layers.batch_normalization(inputs = conv1_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv1_5")
    pool1 = tf.layers.max_pooling2d(inputs = conv1_5, pool_size = 2, strides=2, name = "max_pool1")

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv2")
    conv2_1 = tf.layers.batch_normalization(inputs = conv2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv2_1")
    conv2_2 = tf.layers.conv2d(inputs = conv2_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv2_2")
    conv2_3 = tf.layers.batch_normalization(inputs = conv2_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv2_3")
    conv2_4 = tf.layers.conv2d(inputs = conv2_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv2_4")
    conv2_5 = tf.layers.batch_normalization(inputs = conv2_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv2_5")
    pool2 = tf.layers.max_pooling2d(inputs = conv2_5, pool_size = 2, strides=2, name = "max_pool2")

    conv3 = tf.layers.conv2d(inputs = pool2, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv3")
    conv3_1 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv3_1")
    conv3_2 = tf.layers.conv2d(inputs = conv3_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv3_2")
    conv3_3 = tf.layers.batch_normalization(inputs = conv3_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv3_3")
    conv3_4 = tf.layers.conv2d(inputs = conv3_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv3_4")
    conv3_5 = tf.layers.batch_normalization(inputs = conv3_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv3_5")
    pool3 = tf.layers.max_pooling2d(inputs = conv3_5, pool_size = 2, strides=2, name = "max_pool3")

    conv4 = tf.layers.conv2d(inputs = pool3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv4")
    conv4_1 = tf.layers.batch_normalization(inputs = conv4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv4_1")
    conv4_2 = tf.layers.conv2d(inputs = conv4_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv4_2")
    conv4_3 = tf.layers.batch_normalization(inputs = conv4_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv4_3")
    conv4_4 = tf.layers.conv2d(inputs = conv4_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv4_4")
    conv4_5 = tf.layers.batch_normalization(inputs = conv4_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv4_5")

    conv5 = tf.layers.conv2d(inputs = conv4_5, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv5")
    conv5_1 = tf.layers.batch_normalization(inputs = conv5, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv5_1")
    conv5_2 = tf.layers.conv2d(inputs = conv5_1, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv5_2")
    conv5_3 = tf.layers.batch_normalization(inputs = conv5_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv5_3")
    conv5_4 = tf.layers.conv2d(inputs = conv5_3, filters = 512, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv5_4")
    conv5_5 = tf.layers.batch_normalization(inputs = conv5_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv5_5")

    '''Decoder'''
    up6 = tf.concat([conv5_5, conv4_5], axis = 3)
    conv6 = tf.layers.conv2d(inputs = up6, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv6")
    conv6_1 = tf.layers.batch_normalization(inputs = conv6, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv6_1")
    conv6_2 = tf.layers.conv2d(inputs = conv6_1, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv6_2")
    conv6_3 = tf.layers.batch_normalization(inputs = conv6_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv6_3")
    conv6_4 = tf.layers.conv2d(inputs = conv6_3, filters = 256, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv6_4")
    conv6_5 = tf.layers.batch_normalization(inputs = conv6_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv6_5")

    up7 = tf.keras.layers.UpSampling2D(2)(conv6_5)
    up7 = tf.concat([up7, conv3_5], axis = 3)
    conv7 = tf.layers.conv2d(inputs = up7, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv7")
    conv7_1 = tf.layers.batch_normalization(inputs = conv7, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv7_1")
    conv7_2 = tf.layers.conv2d(inputs = conv7_1, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv7_2")
    conv7_3 = tf.layers.batch_normalization(inputs = conv7_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv7_3")
    conv7_4 = tf.layers.conv2d(inputs = conv7_3, filters = 128, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv7_4")
    conv7_5 = tf.layers.batch_normalization(inputs = conv7_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv7_5")

    up8 = tf.keras.layers.UpSampling2D(2)(conv7_5)
    up8 = tf.concat([up8, conv2_5], axis = 3)
    conv8 = tf.layers.conv2d(inputs = up8, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv8")
    conv8_1 = tf.layers.batch_normalization(inputs = conv8, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv8_1")
    conv8_2 = tf.layers.conv2d(inputs = conv8_1, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv8_2")
    conv8_3 = tf.layers.batch_normalization(inputs = conv8_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv8_3")
    conv8_4 = tf.layers.conv2d(inputs = conv8_3, filters = 64, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv8_4")
    conv8_5 = tf.layers.batch_normalization(inputs = conv8_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv8_5")

    up9 = tf.keras.layers.UpSampling2D(2)(conv8_5)
    up9 = tf.concat([up9, conv1_5], axis = 3)
    conv9 = tf.layers.conv2d(inputs = up9, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv9")
    conv9_1 = tf.layers.batch_normalization(inputs = conv9, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv9_1")
    conv9_2 = tf.layers.conv2d(inputs = conv9_1, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv9_2")
    conv9_3 = tf.layers.batch_normalization(inputs = conv9_2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv9_3")
    conv9_4 = tf.layers.conv2d(inputs = conv9_3, filters = 32, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "ref_conv9_4")
    conv9_5 = tf.layers.batch_normalization(inputs = conv9_4, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "ref_conv9_5")

    decoded =  tf.layers.conv2d(inputs = conv9_5, filters = 3, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.sigmoid, name = "decoded_ref")

    return decoded

def main():


    saver = tf.train.Saver()
    out1 = bootstrap_net(p1)
    out2 = iterative_net(out1,False)
    out3 = iterative_net(out2,True)
    out4 = iterative_net(out3,True)
    out5 = refine_net(out4)
    out6 = tf.add(out5,p1)
    # out5 = refine_net(p1)
    cost = tf.losses.mean_squared_error(p2, out6, weights = 1.0, scope = None)
    optimizer=tf.train.AdamOptimizer(lr).minimize(loss=cost)

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess,"./freeze/model.ckpt")
        im_un1, im_gt1 = get_images()
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0,19):
                img_un = im_un1[i*batch_size:(i+1)*batch_size,:,:,:]
                img_gt = im_gt1[i*batch_size:(i+1)*batch_size,:,:,:]
                c = 0
                _, c,img_gen = sess.run([optimizer,cost,out6], feed_dict={p1:img_un , p2:img_gt, lr: learning_rate})
                epoch_loss += c
                cv2.imwrite("./results/img_{}_{}_gt.png".format(epoch,i),img_gt[8,:,:,0]*255.0)
                cv2.imwrite("./results/img_{}_{}_gen.png".format(epoch,i),img_gen[8,:,:,0]*255.0)
                cv2.imwrite("./results/img_{}_{}_un.png".format(epoch,i),img_un[8,:,:,0]*255.0)
                save_path = saver.save(sess, "./freeze/model.ckpt")
            print(epoch,"="*10,">", epoch_loss)


if(__name__=="__main__"):
    main()
