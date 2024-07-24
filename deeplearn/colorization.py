import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imagenet1k import load_imagenet_1k
from model import ColorNet
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transforms import TransformsFinetune
from utils import rgb_to_ycbcr, ycbcr_to_rgb

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

EPOCHS = 250
LR = 0.001
BATCH_SIZE = 256
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
YCBCR_MEAN = torch.Tensor((22115.68229816, -1714.12480085, 1207.379430890)).view(
    1, 3, 1, 1
)
YCBCR_STD = torch.Tensor((56166673.0373194, 2917696.06388108, 2681980.63707231)).view(
    1, 3, 1, 1
)
PRINT_EVERY = 10
BACKBONE_WEIGHTS = "weights/ycbcr_backbone_unnorm.pth"
writer = SummaryWriter(f"logs/{formatted_time}")


def split_input_label(data):

    ycbcr = rgb_to_ycbcr(data)
    input = torch.clone(ycbcr)
    input[:, 1:, :, :] = 0
    # input = (input - YCBCR_MEAN * 255) / (YCBCR_STD * 255)
    label = ycbcr[:, 1:, :, :]
    return input, label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    YCBCR_MEAN = YCBCR_MEAN.to(device)
    YCBCR_STD = YCBCR_STD.to(device)

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

    model = ColorNet(BACKBONE_WEIGHTS, rgb=False).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = ReduceLROnPlateau(optimizer, "min")
    criterion1 = nn.L1Loss()
    criterion2 = nn.KLDivLoss(reduction="mean")

    best_loss = 1000
    patience = 0

    for epoch in range(EPOCHS):

        model = model.train()
        steps = len(train_loader)
        progress_bar = tqdm(total=steps, desc="Training", position=0)
        for i, data in enumerate(train_loader, 0):
            input, label = split_input_label(data)
            input, label = input.to(device), label.to(device)

            output = model(input)
            loss = criterion1(output.flatten(1), label.flatten(1)) + 2 * criterion2(
                F.softmax(output.flatten(1), dim=1),
                F.softmax(label.flatten(1), dim=1),
            )

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
            loss = criterion1(output.flatten(1), label.flatten(1)) + 2 * criterion2(
                F.softmax(output.flatten(1), dim=1),
                F.softmax(label.flatten(1), dim=1),
            )
            avg_loss += loss.item()
            if i < 3:

                # YCbCr -> RGB
                input = input[0].unsqueeze(0).cpu()
                output = output[0].unsqueeze(0).cpu()
                label = label[0].unsqueeze(0).cpu()

                gt = input.clone()
                gt[:, 1:, :, :] += label
                gt = (ycbcr_to_rgb(gt).squeeze(0).cpu().numpy() * 255).astype(np.uint8)

                out = input.clone()
                out[:, 1:, :, :] += output
                out = (ycbcr_to_rgb(out).squeeze(0).cpu().numpy() * 255).astype(
                    np.uint8
                )

                prt = np.concatenate(
                    (
                        gt,
                        np.concatenate(
                            (gt[:1, :, :], gt[:1, :, :], gt[:1, :, :]), axis=0
                        ),
                        out,
                    ),
                    axis=2,
                )
                writer.add_image(
                    f"Validation/{i}",
                    torch.from_numpy(prt),
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
