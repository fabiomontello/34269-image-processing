import numpy as np
import os
import imageio.v3 as iio
import cv2
from sklearn.metrics import root_mean_squared_error

from transforms import TransformsFinetune


def benchmark(image, imageTrue, wantDeltaE=True):

    R1, G1, B1 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    R2, G2, B2 = imageTrue[:, :, 0], imageTrue[:, :, 1], imageTrue[:, :, 2]
    
    rmse_R = root_mean_squared_error(R1, R2)
    rmse_G = root_mean_squared_error(G1, G2)
    rmse_B = root_mean_squared_error(B1, B2)
    
    rmse_total = (rmse_R + rmse_G + rmse_B) / 3    
    delta_E = deltaE(image, imageTrue)

    return rmse_total, delta_E

#The deltaE function was taken from the last comment in: https://stackoverflow.com/questions/57224007/how-to-compute-the-delta-e-between-two-images-using-opencv
def deltaE(img1, img2, colorspace = cv2.COLOR_RGB2LAB):
    # check the two images are of the same size, else resize the two images
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img1.shape[:2]
    h, w = None, None
    # check the height
    if h1 > h2:
        h = h1
    else:
        h = h2
    #check the width
    if w1 > w2:
        w = w1
    else:
        w = w2
    
    img1 = cv2.resize(img1, (h,w))
    img2 = cv2.resize(img2, (h,w))
    # Convert BGR images to specified colorspace
    img1 = cv2.cvtColor(img1, colorspace)
    img2 = cv2.cvtColor(img2, colorspace)
    # compute the Euclidean distance with pixels of two images 
    return np.mean(np.sqrt(np.sum((img1 - img2) ** 2, axis=-1)))

def getScores(getDelta=True, filepath="output/"):
    transform = TransformsFinetune(train=False, image_size=224)
    files = os.listdir(filepath + "test")
    scores = np.zeros([len(files),2])
    i = 0
    
    for file in files:
        if not file.startswith('.'):# and os.path.isfile(os.path.join(root, file)):
            rgb = iio.imread(filepath + "ycbcr_two_norm/" + file, pilmode="RGB")
            test = iio.imread(filepath + "test/" + file, pilmode="RGB")
            tr_test = transform(test).numpy()
            tr_test = tr_test.transpose(1, 2, 0)
            print(file, "rgb:", rgb.shape, "test:", test.shape, "tr_test", tr_test.shape)
            scores[i] = benchmark(rgb, tr_test)
            print(i, file, scores[i])
            i+=1
    scores = scores[~(scores == 0).all(axis=1)]
    return scores

scores = getScores()
scoresAvg = np.mean(scores, axis=0)
