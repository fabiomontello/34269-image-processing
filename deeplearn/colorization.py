import datetime

import cv2
import numpy as np
import torch
from imagenet1k import load_imagenet_1k
from model import ColorNet
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transforms import TransformsFinetune
from utils import rgb_to_ycbcr

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

EPOCHS = 50
LR = 0.0001
BATCH_SIZE = 172
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
PRINT_EVERY = 10
BACKBONE_WEIGHTS = "weights/ycrcb_backbone.pth"
writer = SummaryWriter(f"logs/{formatted_time}")


def split_input_label(data):
    ycbcr = rgb_to_ycbcr(data)
    input = torch.clone(ycbcr)
    input[:, 1:, :, :] = 0
    label = ycbcr[:, 1:, :, :]
    return input, label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RGB_MEAN = RGB_MEAN.to(device)
    RGB_STD = RGB_STD.to(device)

    train_transform = TransformsFinetune(train=True, image_size=224)
    test_transform = TransformsFinetune(train=False, image_size=224)

    # Load the ImageNet dataset
    train_loader, val_loader, test_loader = load_imagenet_1k(
        data_path=DATA_PATH,
        transform_train=train_transform,
        transform_test=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    model = ColorNet(BACKBONE_WEIGHTS).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, "min")
    criterion = nn.L1Loss()
    best_loss = 1000
    patience = 0

    for epoch in range(EPOCHS):

        model = model.train()
        steps = len(train_loader)
        progress_bar = tqdm(total=steps, desc="Training", position=0)
        for i, data in enumerate(train_loader, 0):
            input, label = split_input_label(data)
            input = input.to(device)
            # label = label.to(device)

            output = model(input)
            loss = criterion(output.cpu(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss/train",
                loss,
                epoch * len(train_loader) + i,
            )
            if (i) % PRINT_EVERY == 0:

                progress_bar.set_postfix({"Loss": f"{(loss):.4f}"})
                progress_bar.update(PRINT_EVERY)

        # evaluate the student model
        model = model.eval()
        avg_loss = 0
        for i, data in enumerate(tqdm(val_loader), 0):
            input, label = split_input_label(data)
            input = input.to(device)
            # label = label.to(device)
            with torch.no_grad():
                output = model(input)
            loss = criterion(output.cpu(), label)
            avg_loss += loss.item()
            if i < 3:
                # YCbCr -> RGB
                input = input[0].cpu()
                output = output[0].cpu()
                label = label[0].cpu()

                gt = input.numpy()
                gt[1:, :, :] += label.numpy()
                gt = np.transpose(gt, (1, 2, 0))
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCR_CB)

                out = input.numpy()
                out[1:, :, :] += output.numpy()
                out = np.transpose(out, (1, 2, 0))
                out = cv2.cvtColor(out, cv2.COLOR_BGR2YCR_CB)

                writer.add_image(
                    f"Validation/{i}",
                    torch.from_numpy(gt).permute(2, 0, 1),
                    epoch,
                )
        avg_loss /= len(val_loader)
        print(f"Epoch: {epoch}, Loss: {avg_loss}")
        scheduler.step(avg_loss)

        writer.add_scalar("Loss/val", avg_loss, epoch)
        patience += 1
        # Save the model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), f"logs/{formatted_time}/color_best.pth")

        if patience > int(EPOCHS * 0.1):
            print("Early stopping")
            break
