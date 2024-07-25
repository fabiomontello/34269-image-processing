import os

import numpy as np
import torch
from imagenet1k import load_imagenet_1k
from model import ColorNet
from PIL import Image
from tqdm import tqdm
from transforms import TransformsFinetune
from utils import rgb_to_ycbcr, ycbcr_to_rgb

BATCH_SIZE = 128
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
YCBCR_MEAN = torch.Tensor((22115.68229816, -1714.12480085, 1207.379430890)).view(
    1, 3, 1, 1
)
YCBCR_STD = torch.Tensor((56166673.0373194, 2917696.06388108, 2681980.63707231)).view(
    1, 3, 1, 1
)
PRINT_EVERY = 10
MODEL_WEIGHTS = (
    "/home/fabmo/works/34269-image-processing/weights/ycbcr_best_two_norm.pth"
)
OUT_DIR = "output/ycbcr_two_norm/"


def split_input_label(data):

    ycbcr = rgb_to_ycbcr(data)
    ycbcr = (ycbcr - YCBCR_MEAN) / (YCBCR_STD)
    input = torch.clone(ycbcr)
    input[:, 1:, :, :] = 0
    label = ycbcr[:, 1:, :, :]
    return input, label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    YCBCR_MEAN = YCBCR_MEAN  # .to(device)
    YCBCR_STD = YCBCR_STD  # .to(device)

    train_transform = TransformsFinetune(train=True, image_size=224)
    test_transform = TransformsFinetune(train=False, image_size=224)

    # Load the ImageNet dataset
    _, _, test_loader = load_imagenet_1k(
        data_path=DATA_PATH,
        transform_train=train_transform,
        transform_test=test_transform,
        batch_size=1,
        num_workers=4,
    )
    model = ColorNet(None, rgb=False).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    for i, data in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input, label = split_input_label(data)
            input, label = input.to(device), label.to(device)

            output = model(input)
            # YCbCr -> RGB
            input = input[0].unsqueeze(0).cpu()
            output = output[0].unsqueeze(0).cpu()
            label = label[0].unsqueeze(0).cpu()

            gt = input.clone()
            gt[:, 1:, :, :] += label
            gt = gt * YCBCR_STD + YCBCR_MEAN
            gt = (ycbcr_to_rgb(gt).squeeze(0).cpu().numpy() * 255).astype(np.uint8)

            out = input.clone()
            out[:, 1:, :, :] += output
            out = out * YCBCR_STD + YCBCR_MEAN
            out = (ycbcr_to_rgb(out).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            file_name = test_loader.dataset.img_list[i].split("/")[-1]
            output = Image.fromarray(out.transpose(1, 2, 0))
            # make dir if it does not exist
            output_dir = f"{OUT_DIR}/{file_name}"
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            output.save(f"{output_dir}")

    print("Data loaded successfully!")
