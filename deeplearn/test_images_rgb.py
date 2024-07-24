import os

import numpy as np
import torch
from imagenet1k import load_imagenet_1k
from model import ColorNet
from PIL import Image
from tqdm import tqdm
from transforms import TransformsFinetune

BATCH_SIZE = 128
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
PRINT_EVERY = 10
MODEL_WEIGHTS = "weights/RGB_best_two.pth"
OUT_DIR = "output/rgb_two/"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RGB_MEAN = RGB_MEAN.to(device)
    RGB_STD = RGB_STD.to(device)

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
    model = ColorNet(None, rgb=True).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    for i, input in tqdm(enumerate(test_loader)):
        input = input.to(device)
        input = (input - RGB_MEAN * 255) / (RGB_STD * 255)

        output = model(input)
        output = (output * 255).cpu().detach().numpy().astype(np.uint8).squeeze(0)
        file_name = test_loader.dataset.img_list[i].split("/")[-1]
        output = Image.fromarray(output.transpose(1, 2, 0))
        # make dir if it does not exist
        output_dir = f"{OUT_DIR}/{file_name}"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        output.save(f"{output_dir}.jpeg")

    print("Data loaded successfully!")
