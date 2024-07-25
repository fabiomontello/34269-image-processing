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

# start time
t0=time()

# crop images around the center
def centerGop(img, cropSze=(224, 224)):
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
            dataset.append(centerGop(img))
    return dataset

# location of dataset
imgDir = '/zhome/ad/a/211839/34269-image-processing/data/imagenet-val/imagenet-val/val/'
trainImg=loadDataset(imgDir)
testDir = '/zhome/ad/a/211839/34269-image-processing/data/imagenet-val/imagenet-val/tst/'
testImg=loadDataset(testDir)

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

# separate channels and normalize values
trainR = np.array([img[:, :, 0] for img in trainImg])/255 # R channel
trainG = np.array([img[:, :, 1] for img in trainImg])/255 # G channel
trainB = np.array([img[:, :, 2] for img in trainImg])/255 # B channel

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
patchR = imgPatch(trainR, sze, mx)
patchG = imgPatch(trainG, sze, mx)
patchB = imgPatch(trainB, sze, mx)

# reshape for dict learning
patchR2D = patchR.reshape(patchR.shape[0], -1)
patchG2D = patchG.reshape(patchG.shape[0], -1)
patchB2D = patchB.reshape(patchB.shape[0], -1)

# number of dict atoms
numComp = 625 # greater than sze[0]*sze[1]
sparseTarget=16

# init DictionaryLearning models
dictR=dictLearn(patchR2D,numComp,sparseTarget)
dictG=dictLearn(patchG2D,numComp,sparseTarget)
dictB=dictLearn(patchB2D,numComp,sparseTarget)

# transform patches using dictionaries
transR = (patchR2D.T@dictR).T
transG = (patchG2D.T@dictG).T
transB = (patchB2D.T@dictB).T


# function to colorize greyscale image using RBG
def colorizeImg(greyImg,dictR, dictB, dictG,patchSze,mxPatches):
    # normalize values
    greyImgRGB=cv2.cvtColor(greyImg, cv2.COLOR_GRAY2RGB)
    greyImgR=greyImgRGB[:,:,0]/255
    greyImgG=greyImgRGB[:,:,1]/255
    greyImgB=greyImgRGB[:,:,2]/255
    # get patches
    patchesR = extract_patches_2d(greyImgR, patchSze, max_patches=mxPatches)
    patchesG = extract_patches_2d(greyImgG, patchSze, max_patches=mxPatches)
    patchesB = extract_patches_2d(greyImgB, patchSze, max_patches=mxPatches)

    # reshape to match dictionary
    reshapedR = patchesB.reshape(patchesR.shape[0], -1)
    reshapedB = patchesB.reshape(patchesB.shape[0], -1)
    reshapedG = patchesB.reshape(patchesG.shape[0], -1)

    # encode sparse representation of R channel using the R dictionary
    coderR = SparseCoder(dictionary=dictR)
    coderG = SparseCoder(dictionary=dictG)
    coderB = SparseCoder(dictionary=dictB)

    # transform RGB channels
    transR = coderR.transform(reshapedR)
    transB = coderG.transform(reshapedB)
    transG = coderB.transform(reshapedG)
    
    # reconstruct patches
    recPatchB = transR @ dictR
    recPatchB = transB @ dictB
    recPatchG = transG @ dictG

    # return to original shape
    recPatchR = recPatchB.reshape(patchesR.shape)
    recPatchB = recPatchB.reshape(patchesB.shape)
    recPatchG = recPatchG.reshape(patchesG.shape)

    # reconstruct channels from patches
    recR = reconstruct_from_patches_2d(recPatchR, greyImg.shape)
    recB = reconstruct_from_patches_2d(recPatchB, greyImg.shape)
    recG = reconstruct_from_patches_2d(recPatchG, greyImg.shape)

    colorImgRGB = np.array([recR,recG,recB]).T*255

    # convert to RGB

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
        imgRGB = testImg[i]
        # convert to greyscale
        greyImg = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        # colorize using dictionary learning
        colorizedImg = colorizeImg(greyImg, transR, transB, transG, sze, 10000000)
        
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
        plt.savefig('./imgRGB'+str(i)+'.png')

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

         # save just the image separately
        cv2.imwrite('./plainImgRGB'+str(i)+'.png',colorizedImg)

test(3)
print('done in %.2f minutes' % ((time() - t0)/60.0))