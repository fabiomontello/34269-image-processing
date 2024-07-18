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


# In[5]:


def dictLearn(signals,atoms,sparse):

    dictionary = dct_dict_1d(
        n_atoms=atoms,
        size=signals.shape[0],
    )
    new_dictionary, errors, iters = train_dict(signals, dictionary, sparsity_target=sparse)
    return new_dictionary


# In[4]:


# load CIFAR-10 dataset
(trainImg, _), (_, _) = cifar10.load_data()

# reduce size of dataset
N = 200
trainSub = trainImg[:N]

# convert to YCrCb
def convert_to_ycrcb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

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

sze = (9, 9)  # size of patches
mx = 100  # max number of patches

# extract patches
patchCr = imgPatch(trainCr, sze, mx)
patchCb = imgPatch(trainCb, sze, mx)
print(patchCr.shape)
# reshape for dict learning
patchCr2D = patchCr.reshape(patchCr.shape[0], -1)
patchCb2D = patchCb.reshape(patchCb.shape[0], -1)

# number of dict atoms
numComp = 100



# In[ ]:





# In[ ]:


# init DictionaryLearning models
#dictCr = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCr=dictLearn(patchCr2D,numpComp,8)


# In[ ]:


#dictCb = DictionaryLearning(n_components=numComp, transform_algorithm='lasso_lars', transform_alpha=1.0, n_jobs=numCores)
dictCb=dictLearn(patchCb2D,numpComp,8)


# In[ ]:


# learn dictionaries and transform patches
transCr = dictCr#.fit(patchCr2D).transform(patchCr2D)


# In[ ]:


transCb = dictCb#.fit(patchCb2D).transform(patchCb2D)


# In[ ]:


# function to colorize greyscale image using YCbCr
def colorizeImg(greyImg, dictCb, dictCr, patchSze,mxPatches):
    # get patches
    patchesCb = imgPatch([greyImg], patchSze, mxPatches)
    patchesCr = imgPatch([greyImg], patchSze, mxPatches)

    # reshape to match dictionary
    reshapedCb = patchesCb.reshape(-1,patchSze[0]*patchSze[1])
    reshapedCr = patchesCr.reshape(-1,patchSze[0]*patchSze[1])
    
    print(patchesCb.shape)
    
    coderCb = SparseCoder(dictionary=dictCb.components_, transform_algorithm='lasso_lars', transform_alpha=1.0)
    coderCr = SparseCoder(dictionary=dictCr.components_, transform_algorithm='lasso_lars', transform_alpha=1.0)

    # transform Cb and Cr channels
    transCb = coderCb.transform(reshapedCb)
    transCr = coderCr.transform(reshapedCr)

    # reconstruct patches
    recPatchCb = np.dot(transCb, dictCb.components_)
    recPatchCr = np.dot(transCr, dictCr.components_)

    # return to original shape
    recPatchCb = recPatchCb.reshape(patchesCb.shape)
    recPatchCr = recPatchCr.reshape(patchesCr.shape)

    # reconstruct channels from patches
    recCb = reconstruct_from_patches_2d(recPatchCb, greyImg.shape)
    recCr = reconstruct_from_patches_2d(recPatchCr, greyImg.shape)

    # combine channels (Y=greyImg)
    colorImg=np.array([greyImg,recCr,recCb]).T
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
        #imgRGB = trainImg[N+i+100]
        imgRGB = cv2.imread('./roald.jpg', cv2.IMREAD_COLOR)
        #print(imgRGB.shape)
        # convert to greyscale
        greyImg = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
        # colorize using dictionary learning
        colorizedImg = colorizeImg(greyImg, dictCb, dictCr, sze, 10000000)
        
        
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
        plt.savefig('img'+str(i)+'.png')



# In[12]:


test(10)


# Num Images = 100<br>
# Dict Atoms = 100<br>
# Patch Size = (3,3)<br>
# started around 9:35am<br>
# ended 9:57am<br>
# 
# Num Images = 100<br>
# Dict Atoms = 100<br>
# Patch Size = (9,9)<br>
# started 10:09 am<br>
# ended 11:46am<br>
# 

# In[58]:


from imagenet1k import load_imagenet_1k
from transforms import Transforms

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 16
DATA_PATH = "data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406))
RGB_STD = torch.Tensor((0.229, 0.224, 0.225))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transform = Transforms(train=True, image_size=224)
    test_transform = Transforms(train=False, image_size=224)
    train_loader, val_loader, test_loader = load_imagenet_1k(
        data_path=DATA_PATH,
        transform_train=train_transform,
        transform_test=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )


# In[ ]:


from sktime.performance_metrics.forecasting import mean_relative_absolute_error as mrae
from sklearn.metrics import root_mean_squared_error as rmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as pgnr



print(f'RMSE: {rmse(imgRGB,colorizedImg):.2f}')
print(f'MRAE: {mrae(imgRGB,colorizedImg):.2f}')
print(f'SSIM: {ssim(imgRGB,colorizedImg):.2f}')
print(f'PGNR: {pgnr(imgRGB,colorizedImg):.2f}')


# In[ ]:




