import numpy as np
import cv2
# for patch extraction from images
from sklearn.feature_extraction.image import extract_patches_2d 
# for image reconstruction using patches
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
import matplotlib.pyplot as plt # for depicting images
from time import time # used to time model
from PIL import Image # for image cropping
import os # for file reading
# dictionary learning algorithms that run on GPU using CUDA
# https://github.com/mukheshpugal/dictlearn_gpu 
from dictlearn_gpu import train_dict 
from dictlearn_gpu.utils import dct_dict_1d
# for parallel computations on multiple CPU cores
import multiprocessing
from joblib import Parallel, delayed

# start time
t0=time()

# crop images around the center
def centerCrop(img, cropSze=(224, 224)):
    # original size
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
    
    # convert to PIL image to crop
    pilImg=Image.fromarray(img)
    cropped=pilImg.crop((left, top, right, bottom))
    # convert back to numpy array
    return np.asarray(cropped)

# load data from directory
def loadDataset(dir):
    # initialize
    dataset=[]
    # find all files in given directory
    for file in os.listdir(dir):
        # read files as images using opencv
        img = cv2.imread(os.path.join(dir,file), cv2.IMREAD_COLOR)
        # check that the object is not empty
        if img.all() != None:
            # crop then append to dataset
            dataset.append(centerCrop(img))
    return dataset

# location of dataset
imgDir = '/zhome/ad/a/211839/34269-image-processing/data/imagenet-val/imagenet-val/val/'
trainImg=loadDataset(imgDir)

# create and learn dictionary
def dictLearn(signals,atoms,sparse):
    # create dictionary with given parameters
    dictionary = dct_dict_1d(
        n_atoms=atoms,
        size=signals.shape[0],
    )
    # train the dictionary with signals
    updated_dictionary, errors, iters = train_dict(signals, dictionary, sparsity_target=sparse)
    return updated_dictionary

# convert to YCrCb
def convert_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

# use more CPU cores for faster training
trainImgYCrCb = [convert_to_ycrcb(img) for img in trainImg]
#numCores = multiprocessing.cpu_count()
#trainImgYCrCb = Parallel(n_jobs=numCores)(delayed(convert_to_ycrcb)(img) for img in trainImg)

# separate channels and normalize values
trainY = np.array([img[:, :, 0] for img in trainImgYCrCb])/255 # Y channel
trainCr = np.array([img[:, :, 1] for img in trainImgYCrCb])/255 # Cr channel
trainCb = np.array([img[:, :, 2] for img in trainImgYCrCb])/255 # Cb channel

# extract patches (done in parallel)
def imgPatch(imgs, szePatch, maxPatch):
    def extract(img):
        return extract_patches_2d(img, szePatch, max_patches=maxPatch)
    #patches = Parallel(n_jobs=numCores)(delayed(extract)(img) for img in imgs)
    patches = [extract(img) for img in imgs]

    return np.concatenate(patches, axis=0)

# size of patches
sze = (16, 16)
# max number of patches
mx = 10000000  

# extract patches
patchY = imgPatch(trainY, sze, mx)
patchCr = imgPatch(trainCr, sze, mx)
patchCb = imgPatch(trainCb, sze, mx)

# reshape for dict learning
patchY2D = patchY.reshape(patchY.shape[0], -1)
patchCr2D = patchCr.reshape(patchCr.shape[0], -1)
patchCb2D = patchCb.reshape(patchCb.shape[0], -1)

# number of dict atoms
numComp = 625 # greater than sze[0]*sze[1]
sparseTarget=16

# init DictionaryLearning models
dictY=dictLearn(patchY2D,numComp,sparseTarget)
dictCr=dictLearn(patchCr2D,numComp,sparseTarget)
dictCb=dictLearn(patchCb2D,numComp,sparseTarget)

# transform patches using dictionaries
transY = (patchY2D.T@dictY).T
transCr = (patchCr2D.T@dictCr).T
transCb = (patchCb2D.T@dictCb).T


# function to colorize greyscale image using YCbCr
def colorizeImg(greyImg,dictY, dictCb, dictCr,patchSze,mxPatches):
    # normalize values
    greyImg=greyImg/255
    # get patches
    patchesCb = extract_patches_2d(greyImg, patchSze, max_patches=mxPatches)
    patchesCr = patchesCb 
    print(patchesCb.shape)

    # reshape to match dictionary
    reshapedCb = patchesCb.reshape(patchesCb.shape[0], -1)
    reshapedCr = reshapedCb 

    # encode sparse representation of Y channel using the Y dictionary
    coderY = SparseCoder(dictionary=dictY)

    # transform Cb and Cr channels
    transCb = coderY.transform(reshapedCb)
    transCr = coderY.transform(reshapedCr)

    # reconstruct patches
    recPatchCb = transCb @ dictCb
    recPatchCr = transCr @ dictCr

    # return to original shape
    recPatchCb = recPatchCb.reshape(patchesCb.shape)
    recPatchCr = recPatchCr.reshape(patchesCr.shape)

    # reconstruct channels from patches
    recCb = reconstruct_from_patches_2d(recPatchCb, greyImg.shape)
    recCr = reconstruct_from_patches_2d(recPatchCr, greyImg.shape)

    # combine channels (Y=greyImg)
    colorImg = np.array([greyImg,recCr,recCb]).T*255

    # convert to RGB
    colorImgRGB = cv2.cvtColor(colorImg.astype(np.uint8), cv2.COLOR_YCrCb2)

    #plt.imshow(colorImg)
    #plt.savefig('./ycrcb.png')

    # return to original orientation
    colorImgRGB = np.fliplr(colorImgRGB)
    colorImgRGB = np.rot90(colorImgRGB)

    return colorImgRGB

# perform x number of tests
def test(x):
    for i in range(x):
        # test image
        N = 1
        imgRGB = trainImg[N+i]
        # convert to greyscale
        greyImg = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        # colorize using dictionary learning
        colorizedImg = colorizeImg(greyImg, transY, transCb, transCr, sze, 10000000)
        
        # plot ground truth image
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(imgRGB)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # plot greyscale image
        axes[1].imshow(greyImg,cmap='grey')
        axes[1].set_title('Greyscale Image')
        axes[1].axis('off')
        
        # plot colorized image
        axes[2].imshow(colorizedImg, vmin=0, vmax=255)
        axes[2].set_title('Recolorized Image')
        axes[2].axis('off')
        plt.savefig('./img'+str(i)+'.png')

        # plot red channel
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(colorizedImg[:, :, 0],cmap='Reds', vmin=0, vmax=255)
        axes[0].set_title('R')
        axes[0].axis('off')
        
        # plot green channel
        axes[1].imshow(colorizedImg[:, :, 1],cmap='Greens', vmin=0, vmax=255)
        axes[1].set_title('G')
        axes[1].axis('off')
        
        # plot blue channel
        axes[2].imshow(colorizedImg[:, :, 2],cmap='Blues', vmin=0, vmax=255)
        axes[2].set_title('B')
        axes[2].axis('off')
        plt.savefig('./rgb'+str(i)+'.png')

test(20)
print('done in %.2f minutes' % ((time() - t0)/60.0))