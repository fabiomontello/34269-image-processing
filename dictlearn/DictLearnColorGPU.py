#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# In[5]:

t0=time()

def dictLearn(signals,atoms,sparse):

    dictionary = dct_dict_1d(
        n_atoms=atoms,
        size=signals.shape[0],
    )
    updated_dictionary, errors, iters = train_dict(signals, dictionary, sparsity_target=sparse)
    return updated_dictionary


# In[4]:


# load CIFAR-10 dataset
(trainImg, _), (_, _) = cifar10.load_data()

# reduce size of dataset
N = 10
trainSub = trainImg[:N]

# convert to YCrCb
def convert_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

numCores = multiprocessing.cpu_count() # trying to use more CPU cores for faster training
trainImgYCrCb = Parallel(n_jobs=numCores)(delayed(convert_to_ycrcb)(img) for img in trainSub)

# separate channels (Y not needed because it will remain unchanged)
trainCr = np.array([img[:, :, 1] for img in trainImgYCrCb]) # Cr channel
trainCb = np.array([img[:, :, 2] for img in trainImgYCrCb]) # Cb channel

# extract patches (parallel)
def imgPatch(imgs, szePatch, maxPatch):
    def extract(img):
        return extract_patches_2d(img, szePatch, max_patches=maxPatch)
    patches = Parallel(n_jobs=numCores)(delayed(extract)(img) for img in imgs)
    return np.concatenate(patches, axis=0)

sze = (4, 4)  # size of patches
mx = 10000000  # max number of patches

# extract patches
patchCr = imgPatch(trainCr, sze, mx)
patchCb = imgPatch(trainCb, sze, mx)
print('patchCb')
print(patchCb.shape)
# reshape for dict learning
patchCr2D = patchCr.reshape(patchCr.shape[0], -1)
patchCb2D = patchCb.reshape(patchCb.shape[0], -1)
print('patchCb2D')
print(patchCb2D.shape)
# number of dict atoms
numComp = 100#sze[0]*sze[1]#100

# init DictionaryLearning models
#dictCr = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCr=dictLearn(patchCr2D,numComp,16)

#dictCb = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCb=dictLearn(patchCb2D,numComp,16)


# In[ ]:


# learn dictionaries and transform patches
transCr = dictCr#.fit(patchCr2D).transform(patchCr2D)


# In[ ]:


transCb = dictCb#.fit(patchCb2D).transform(patchCb2D)


# In[ ]:


# function to colorize greyscale image using YCbCr
def colorizeImg(greyImg, dictCb, dictCr,patchSze,mxPatches,numComp):
    # get patches
    patchesCb = imgPatch([greyImg], patchSze, mxPatches)
    patchesCr = imgPatch([greyImg], patchSze, mxPatches)

    # reshape to match dictionary
    reshapedCb = patchesCb.reshape(patchesCb.shape[0], -1)
    reshapedCr = patchesCr.reshape(patchesCr.shape[0], -1)
    #transCb=reshapedCb
    #transCr=reshapedCr

    
    coderCb = SparseCoder(dictionary=dictCb,transform_n_nonzero_coefs=numComp)#, transform_algorithm='lasso_lars', transform_alpha=10.0)
    coderCr = SparseCoder(dictionary=dictCr,transform_n_nonzero_coefs=numComp)#, transform_algorithm='lasso_lars', transform_alpha=10.0)

    print('patchesCb')
    print(patchesCb.shape)
    print('reshapedCb')
    print(reshapedCb.shape)
    print('dictCb')
    print(dictCb.shape)

    # transform Cb and Cr channels
    transCb = coderCb.transform(reshapedCb)
    transCr = coderCr.transform(reshapedCr)
    print('transCb')
    print(transCb.shape)
    # reconstruct patches
    recPatchCb = np.dot(transCb, dictCb)
    recPatchCr = np.dot(transCr, dictCr)
    print('recPatchCb')
    print(recPatchCb.shape)

    # return to original shape
    recPatchCb = recPatchCb.reshape(patchesCb.shape)
    recPatchCr = recPatchCr.reshape(patchesCr.shape)
    print(recPatchCb.shape)

    # reconstruct channels from patches
    recCb = reconstruct_from_patches_2d(recPatchCb, greyImg.shape)
    recCr = reconstruct_from_patches_2d(recPatchCr, greyImg.shape)

    # combine channels (Y=greyImg)
    colorImg=np.array([greyImg,recCr,recCb]).T
    print('colorImg')
    print(colorImg.shape)
    # convert to RGB
    colorImgRGB = cv2.cvtColor(colorImg.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    colorImgRGB=np.fliplr(colorImgRGB)
    colorImgRGB=np.rot90(colorImgRGB)
    return colorImgRGB


# In[11]:


from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

def test(x):
    for i in range(x):
        # test image
        imgRGB = trainImg[N+i+100]
        #imgRGB = cv2.imread('./roald.jpg', cv2.IMREAD_COLOR)
        #print(imgRGB.shape)
        # convert to greyscale
        greyImg = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        # colorize using dictionary learning
        colorizedImg = colorizeImg(greyImg, dictCb, dictCr, sze, 10000000,numComp)
        
        
        # plot images
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(imgRGB)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(greyImg,cmap='grey')
        axes[1].set_title('Greyscale Image')
        axes[1].axis('off')
        
        axes[2].imshow(colorizedImg)
        axes[2].set_title('Recolorized Image')
        axes[2].axis('off')
        plt.savefig('./img'+str(i)+'.png')



# In[12]:


test(1)