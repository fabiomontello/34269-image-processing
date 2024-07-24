import colour
import numpy as np
import os
import imageio.v3 as iio
from skimage import color
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_relative_absolute_error

from transforms import TransformsFinetune
#from PIL import Image
#import matplotlib.pyplot as plt


def benchmark(image, imageTrue, wantDeltaE=True):
    #Split images into RGB
    R1, G1, B1 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    R2, G2, B2 = imageTrue[:, :, 0], imageTrue[:, :, 1], imageTrue[:, :, 2]

    #Calculate rmse for each channel
    rmse_R = root_mean_squared_error(R1, R2)
    rmse_G = root_mean_squared_error(G1, G2)
    rmse_B = root_mean_squared_error(B1, B2)
    rmse_total = (rmse_R + rmse_G + rmse_B) / 3

    #Calculate mrae for each channel
    #Need a benchmark image for MRAE, using the ground image as benchmark leads to essentially dividing by 0
    mrae_R = mean_relative_absolute_error(R2, R1, y_pred_benchmark=R2) 
    mrae_G = mean_relative_absolute_error(G2, G1, y_pred_benchmark=G2)
    mrae_B = mean_relative_absolute_error(B2, B1, y_pred_benchmark=B2)
    mrae_total = (mrae_R + mrae_G + mrae_B) / 3

    
    if wantDeltaE == True: #Can run function with only the two fast metric
        imageLAB = color.rgb2lab(image)
        imageTrueLAB = color.rgb2lab(imageTrue)
        delta_E_pixels = np.zeros([imageLAB.shape[0], imageLAB.shape[1]])
        #Calculate delta_E for each pixel in the images, 
        for i in range(imageLAB.shape[0]):
            for j in range(imageLAB.shape[1]):
                delta_E_pixels[i,j] = colour.delta_E(imageLAB[i,j,:], imageTrueLAB[i,j,:], method="CIE 2000")
        #Average of all delta_E values
        delta_E = np.mean(delta_E_pixels)
    else:
        delta_E = None
    
    return rmse_total, mrae_total, delta_E

def getScores(getDelta=True, filepath="output/"):
    #Transform turns test folder images to same resolution as rgb folder
    transform = TransformsFinetune(train=False, image_size=224)
    files = os.listdir(filepath + "test")
    scores = np.zeros([len(files), 3])
    i = 0
    
    for file in files:
        if not file.startswith('.'): #Ignore .DS_Store file that is sometimes in folders
            #Takes each file in test and finds the 
            rgb = iio.imread(filepath + "rgb/" + file, pilmode="RGB")
            test = iio.imread(filepath + "test/" + file, pilmode="RGB")
            tr_test = transform(test).numpy()
            tr_test = tr_test.transpose(1, 2, 0)
            #With the two images loaded and test in the right form, gets benchmarks. Passes whether to get the delta_E or not to the benchmark function
            scores[i] = benchmark(rgb, tr_test, getDelta)
            i+=1
    scores = scores[~(scores == 0).all(axis=1)] #Deletes rows with 0s if some files were ignored
    return scores

#Runs the function once, goes through both folders
scores = getScores()
