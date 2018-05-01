# Copyright 2013-2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
import numpy as np
import os
import cv2
import pydicom
import numpy as np
from skimage.measure import compare_ssim as ssim

def readcfl():
    x = np.zeros((320,320,256))
    for k in range(256):
        for i in range(0,320,2):
            for j in range(320):
                x[i][j][k] = 1
                
    for i in range(1,20):       
        if(i==12):
            continue
        path = "/home/dlagroup5/Project/Data/Kspace/P{}".format(i)
        name = "kspace"
        h = open(os.path.join(path,"kspace.hdr"), "r")
        h.readline()
        l = h.readline()
        h.close()
        dims = [int(i) for i in l.split()]
        n = np.prod(dims)
        dims_prod = np.cumprod(dims)
        dims = dims[:np.searchsorted(dims_prod, n)+1]
        d = open(os.path.join(path,"kspace.cfl"), "r")
        a = np.fromfile(d, dtype=np.complex64, count=n);
        d.close()
        a = np.reshape(a,dims, order='F')
        print(a.shape)

        ct = 0
        temp1 = np.multiply(a[:,:,:,0],x)
        temp2 = a[:,:,:,0]
        for j in range(1,4):
            img_1 = a[:,:,:,j]
            img_1 = np.multiply(img_1, x)
            #img1 = np.fft.fftshift(np.fft.ifftn(img_))
            temp_1 = np.concatenate((temp1,img_1), axis = 2)
            temp1 = temp_1
            
            img_2 = a[:,:,:,j]
            #img2 = np.fft.fftshift(np.fft.ifftn(img_))
            temp_2 = np.concatenate((temp2,img_2), axis = 2)
            temp2 = temp_2
            print(j)
            
        np.save("/home/dlagroup5/Project/undersampled/network_codes/data/input{}.npy".format(i),temp1)
        np.save("/home/dlagroup5/Project/undersampled/network_codes/data/output{}.npy".format(i),temp2)





def load_train_data():
    dat = np.load("train_data. npy")
    img = np.load("train_gt.npy")
    return dat

if __name__=="__main__":
    readcfl()
