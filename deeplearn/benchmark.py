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

    R1, G1, B1 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    R2, G2, B2 = imageTrue[:, :, 0], imageTrue[:, :, 1], imageTrue[:, :, 2]
    
    rmse_R = root_mean_squared_error(R1, R2)
    rmse_G = root_mean_squared_error(G1, G2)
    rmse_B = root_mean_squared_error(B1, B2)
    rmse_total = (rmse_R + rmse_G + rmse_B) / 3
    
    #Need a benchmark image for MRAE, using the ground image as benchmark leads to essentially dividing by 0
    mrae_R = mean_relative_absolute_error(R2, R1, y_pred_benchmark=R2) 
    mrae_G = mean_relative_absolute_error(G2, G1, y_pred_benchmark=G2)
    mrae_B = mean_relative_absolute_error(B2, B1, y_pred_benchmark=B2)
    mrae_total = (mrae_R + mrae_G + mrae_B) / 3
    
    if wantDeltaE == True:
        imageLAB = color.rgb2lab(image)
        imageTrueLAB = color.rgb2lab(imageTrue)
        delta_E_pixels = np.zeros([imageLAB.shape[0], imageLAB.shape[1]])
        for i in range(imageLAB.shape[0]):
            for j in range(imageLAB.shape[1]):
                delta_E_pixels[i,j] = colour.delta_E(imageLAB[i,j,:], imageTrueLAB[i,j,:], method="CIE 2000")
        delta_E = np.mean(delta_E_pixels)
    else:
        delta_E = None
    
    return rmse_total, mrae_total, delta_E

def getScores(getDelta=True, filepath="output/"):
    transform = TransformsFinetune(train=False, image_size=224)
    files = os.listdir(filepath + "test")
    scores = np.zeros([len(files), 3])
    i = 0
    
    for file in files:
        if not file.startswith('.'):# and os.path.isfile(os.path.join(root, file)):
            rgb = iio.imread(filepath + "rgb/" + file, pilmode="RGB")
            test = iio.imread(filepath + "test/" + file, pilmode="RGB")
            if len(test.shape) < 3:
                test = [test, test, test]
            tr_test = transform(test).numpy()
            tr_test = tr_test.transpose(1, 2, 0)
            print(file, "rgb:", rgb.shape, "test:", test.shape, "tr_test", tr_test.shape)
            scores[i] = benchmark(rgb, tr_test, getDelta)
            print(i, file, scores[i])
            i+=1
    scores = scores[~(scores == 0).all(axis=1)]
    return scores

scores = getScores()
