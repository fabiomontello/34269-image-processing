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
trainY = np.array([img[:, :, 0] for img in trainImgYCrCb])/255 # Y channel
trainCr = np.array([img[:, :, 1] for img in trainImgYCrCb])/255 # Cr channel
trainCb = np.array([img[:, :, 2] for img in trainImgYCrCb])/255 # Cb channel

# extract patches (parallel)
def imgPatch(imgs, szePatch, maxPatch):
    def extract(img):
        return extract_patches_2d(img, szePatch, max_patches=maxPatch)
    patches = Parallel(n_jobs=numCores)(delayed(extract)(img) for img in imgs)
    return np.concatenate(patches, axis=0)

sze = (12, 12)  # size of patches
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
numComp = sze[0]*sze[1]#100
sparseTarget=numComp
# init DictionaryLearning models
dictY=dictLearn(patchY2D,numComp,sparseTarget)
#dictCr = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCr=dictLearn(patchCr2D,numComp,sparseTarget)

#dictCb = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCb=dictLearn(patchCb2D,numComp,sparseTarget)


# In[ ]:


# learn dictionaries and transform patches
transCr = dictCr#.fit(patchCr2D).transform(patchCr2D)


# In[ ]:


transCb = dictCb#.fit(patchCb2D).transform(patchCb2D)


# In[ ]:


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

    coderY = SparseCoder(dictionary=dictY)
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
    transY = coderY.transform(reshapedY,positive_code=True)
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
    recY = reconstruct_from_patches_2d(recPatchY, greyImg.shape)
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
    colorImgRGB=np.fliplr(colorImgRGB)
    colorImgRGB=np.rot90(colorImgRGB)
    return colorImgRGB


# In[11]:


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
        colorizedImg = colorizeImg(greyImg,dictY, dictCb, dictCr, sze, 10000000,numComp)
        
        
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
print('done in %.2f minutes' % ((time() - t0)/60.0))