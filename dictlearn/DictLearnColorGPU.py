#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import DictionaryLearning
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import multiprocessing
from dictlearn_gpu import train_dict
from dictlearn_gpu.utils import dct_dict_1d
from time import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import sparse_encode
from PIL import Image 
import glob
import os


t0=time()

# load CIFAR-10 dataset
#(trainImg, _), (_, _) = cifar10.load_data()

# reduce size of dataset
N = 5000
#trainSub = trainImg[:N]



def centerCrop(img, cropSze=(256, 256)):
    # original size
    #print(img)
    #print(img.shape)
    w = img.shape[0] 
    h = img.shape[1] 
    # new size
    croppedW = cropSze[0] 
    croppedH = cropSze[1]
    # find center
    left = (w - croppedW) // 2
    top = (h - croppedH) // 2
    right = (w + croppedW) // 2
    bottom = (h + croppedH) // 2
    
    pilImg=Image.fromarray(img)
    cropped=pilImg.crop((left, top, right, bottom))
    return np.asarray(cropped)

def loadDataset(dir):
    dataset=[]
    for file in os.listdir(dir):
        img = cv2.imread(os.path.join(dir,file), cv2.IMREAD_COLOR)
        #print(img.shape)
        if img.all() != None:
                dataset.append(centerCrop(img))
    return dataset

imgDir = '../data/imagenet-val/imagenet-val/val/'
trainSub=loadDataset(imgDir)

#trainSub = [cv2.imread(file) for file in files]

#creating a collection with the available images
#col = images#load_images_from_folder(imgDir)
#trainSub = np.array([cv2.imread('./testImg.JPEG')])#/34269-image-processing/data/imagenet-val/imagenet-val/val/ILSVRC2012_val_00000019.JPEG',cv2.IMREAD_COLOR)])
print(trainSub.shape)
def dictLearn(signals,atoms,sparse):
    dictionary = dct_dict_1d(
        n_atoms=atoms,
        size=signals.shape[0],
    )
    updated_dictionary, errors, iters = train_dict(signals, dictionary, sparsity_target=sparse)
    return updated_dictionary

# convert to YCrCb
def convert_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

numCores = multiprocessing.cpu_count() # trying to use more CPU cores for faster training
trainImgYCrCb = Parallel(n_jobs=numCores)(delayed(convert_to_ycrcb)(img) for img in trainSub)

# separate channels (Y not needed because it will remain unchanged)
trainY = np.array([img[:, :, 0] for img in trainImgYCrCb])/255 # Y channel
trainCr = np.array([img[:, :, 1] for img in trainImgYCrCb])/255 # Cr channel
trainCb = np.array([img[:, :, 2] for img in trainImgYCrCb])/255 # Cb channel

# extract patches (parallel)
def imgPatch(imgs, szePatch, maxPatch):
    def extract(img):
        return extract_patches_2d(img, szePatch, max_patches=maxPatch)
    patches = Parallel(n_jobs=numCores)(delayed(extract)(img) for img in imgs)
    return np.concatenate(patches, axis=0)

sze = (16, 16)  # size of patches
mx = 10000000  # max number of patches

# extract patches
patchY = imgPatch(trainY, sze, mx)
patchCr = imgPatch(trainCr, sze, mx)
patchCb = imgPatch(trainCb, sze, mx)
print('patchCb')
print(patchCb.shape)
# reshape for dict learning
#print(patchCb)
patchY2D = patchY.reshape(patchY.shape[0], -1)
patchCr2D = patchCr.reshape(patchCr.shape[0], -1)
patchCb2D = patchCb.reshape(patchCb.shape[0], -1)
print('patchCb2D')
print(patchCb2D.shape)
#print(patchCb2D)
# number of dict atoms
numComp = 625#sze[0]*sze[1]#100
sparseTarget=16#numComp
# init DictionaryLearning models
dictY=dictLearn(patchY2D,numComp,sparseTarget)
#dictCr = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCr=dictLearn(patchCr2D,numComp,sparseTarget)
#dictCb = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCb=dictLearn(patchCb2D,numComp,sparseTarget)


# learn dictionaries and transform patches
transY = (patchY2D.T@dictY).T
transCr = (patchCr2D.T@dictCr).T#.fit(patchCr2D).transform(patchCr2D)
print('transCr')
print(transCr.shape)
transCb = (patchCb2D.T@dictCb).T#.fit(patchCb2D).transform(patchCb2D)


# function to colorize greyscale image using YCbCr
def colorizeImg(greyImg,dictY, dictCb, dictCr,patchSze,mxPatches,numComp):
    # get patches
    greyImg=greyImg/255
    patchesCb = imgPatch([greyImg], patchSze, mxPatches)
    patchesCr = patchesCb #imgPatch([greyImg], patchSze, mxPatches)
    patchesY=patchesCb

    # reshape to match dictionary
    reshapedCb = patchesCb.reshape(patchesCb.shape[0], -1)
    reshapedCr = reshapedCb #patchesCr.reshape(patchesCr.shape[0], -1)
    reshapedY=reshapedCb
    #transCb=reshapedCb
    #transCr=reshapedCr

    coderY = SparseCoder(dictionary=dictY)#,positive_code=True)
    #coderCb = SparseCoder(dictionary=dictCb)#,transform_n_nonzero_coefs=patchSze[0]*patchSze[1])#, transform_algorithm='lasso_lars', transform_alpha=10.0)
    #coderCr = SparseCoder(dictionary=dictCr)#,transform_n_nonzero_coefs=patchSze[0]*patchSze[1])#, transform_algorithm='lasso_lars', transform_alpha=10.0)
    #coder = sparse_encode(reshapedY, dictY)

    print('patchesCb')
    print(patchesCb.shape)
    print('reshapedCb')
    print(reshapedCb.shape)
    print('dictCb')
    print(dictCb.shape)

    # transform Cb and Cr channels
    transY = coderY.transform(reshapedY)
    transCb = coderY.transform(reshapedCb)#transY#coderCb.transform(reshapedCb)
    transCr = coderY.transform(reshapedCr)#transY#coderCr.transform(reshapedCr)
    print('transCb')
    print(transCb.shape)
    # reconstruct patches
    recPatchY = transY@dictY#np.dot(transY, dictY)
    recPatchCb = transCb@dictCb#np.dot(transCb, dictCb)
    recPatchCr = transCr@dictCr#np.dot(transCr, dictCr)
    print('recPatchCb')
    print(recPatchCb.shape)

    # return to original shape
    recPatchY = recPatchY.reshape(patchesY.shape)
    recPatchCb = recPatchCb.reshape(patchesCb.shape)
    recPatchCr = recPatchCr.reshape(patchesCr.shape)
    print(recPatchCb.shape)
    print(recPatchCb)

    # reconstruct channels from patches
    recY = greyImg#reconstruct_from_patches_2d(recPatchY, greyImg.shape)
    recCb = reconstruct_from_patches_2d(recPatchCb, greyImg.shape)
    recCr = reconstruct_from_patches_2d(recPatchCr, greyImg.shape)
    print('recCb')
    print(recCb)
    # combine channels (Y=greyImg)
    colorImg=np.array([recY,recCr,recCb]).T*255
    print('colorImg')
    print(colorImg.shape)
    # convert to RGB
    colorImgRGB=cv2.cvtColor(colorImg.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    plt.imshow(colorImg)
    plt.savefig('./ycrcb.png')
    colorImgRGB=np.fliplr(colorImgRGB)
    colorImgRGB=np.rot90(colorImgRGB)
    return colorImgRGB


from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def test(x):
    for i in range(x):
        # test image
        imgRGB = trainImg[i]#N+i+100]
        #imgRGB = cv2.imread('./roald.jpg', cv2.IMREAD_COLOR)
        #print(imgRGB.shape)
        # convert to greyscale
        greyImg = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        # colorize using dictionary learning
        #imgs=cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCrCb)
        #colorizedImg = cv2.cvtColor(imgs, cv2.COLOR_YCrCb2RGB)#
        colorizedImg=colorizeImg(greyImg,transY, transCb, transCr, sze, 10000000,numComp)
        
        
        # plot images
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(imgRGB)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(greyImg,cmap='grey')
        axes[1].set_title('Greyscale Image')
        axes[1].axis('off')
        
        axes[2].imshow(colorizedImg, vmin=0, vmax=255)
        axes[2].set_title('Recolorized Image')
        axes[2].axis('off')
        plt.savefig('./img'+str(i)+'.png')

        r=imgRGB[0]
        g=imgRGB[1]
        b=imgRGB[2]
        empt=np.zeros_like(r)

                # plot images
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(colorizedImg[:, :, 0],cmap='grey', vmin=0, vmax=255)
        axes[0].set_title('R')
        axes[0].axis('off')
        
        axes[1].imshow(colorizedImg[:, :, 1],cmap='grey', vmin=0, vmax=255)
        axes[1].set_title('G')
        axes[1].axis('off')
        
        axes[2].imshow(colorizedImg[:, :, 2],cmap='grey', vmin=0, vmax=255)
        axes[2].set_title('B')
        axes[2].axis('off')
        plt.savefig('./rgb'+str(i)+'.png')



test(3)
print('done in %.2f minutes' % ((time() - t0)/60.0))