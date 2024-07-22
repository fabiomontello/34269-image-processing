import torch
import torch.cuda
import torch.nn as nn
from skimage.color import (
    hed2rgb,
    hsv2rgb,
    lab2rgb,
    rgb2hed,
    rgb2hsv,
    rgb2lab,
    rgb2xyz,
    rgb2ycbcr,
    rgb2yuv,
    xyz2rgb,
    ycbcr2rgb,
    yuv2rgb,
)
from torch.autograd import Variable
import colour
import numpy as np
import os
import imageio.v3 as iio
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_relative_absolute_error


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Assuming the model is a nn.Sequential, remove the last layer (head)
        # This is a generic approach; the actual implementation may vary based on the model architecture
        self.features = nn.Sequential(*list(model.children()))

    def forward(self, x):
        x = self.features(x)
        return x


#### The following code for color conversion of pythorch images has been taken from https://github.com/jorge-pessoa/pytorch-colors/blob/master/tests/unit_tests.py


def _convert(input_, type_):
    return {
        "float": input_.float(),
        "double": input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type="", out_type=""):
    def apply_transform(input_):
        to_squeeze = input_.dim() == 3
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)

    return apply_transform


def _generic_transform_sk_3d(transform, in_type="", out_type=""):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)

    return apply_transform


# --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type="double", out_type="float")
# --- YUV ---
rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# --- YCbCr ---
rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type="double", out_type="float")
# --- HSV ---
rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
# --- XYZ ---
rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type="double", out_type="float")
# --- HED ---
rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type="double", out_type="float")


def err(type_):
    raise NotImplementedError("Color space conversion %s not implemented yet" % type_)


def convert(input_, type_):
    return {
        "rgb2lab": rgb_to_lab(input_),
        "lab2rgb": lab_to_rgb(input_),
        "rgb2yuv": rgb_to_yuv(input_),
        "yuv2rgb": yuv_to_rgb(input_),
        "rgb2xyz": rgb_to_xyz(input_),
        "xyz2rgb": xyz_to_rgb(input_),
        "rgb2hsv": rgb_to_hsv(input_),
        "hsv2rgb": hsv_to_rgb(input_),
        "rgb2ycbcr": rgb_to_ycbcr(input_),
        "ycbcr2rgb": ycbcr_to_rgb(input_),
    }.get(type_, err(type_))

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
        delta_E_pixels = np.zeros([image.shape[0], image.shape[1]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                delta_E_pixels[i,j] = colour.delta_E(image[i,j,:], imageTrue[i,j,:], method="CIE 2000")
        delta_E = np.mean(delta_E_pixels)
    else:
        delta_E = None
    return rmse_total, mrae_total, delta_E

def getScores(getDelta=True):
    files = os.listdir("test")
    scores = np.zeros([len(files)+5, 3])
    i = 0
    for file in files:
        if not file.startswith('.'):# and os.path.isfile(os.path.join(root, file)):
            scores[i] = benchmark(iio.imread("test/" + file), iio.imread("val/" + file), getDelta)
            i+=1
    scores = scores[~(scores == 0).all(axis=1)]
    return scores
