import datetime

import numpy as np
import torch
import torchvision.transforms as transforms
from imagenet1k import load_imagenet_1k
from model import ColorNet
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transforms import TransformsFinetune

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

EPOCHS = 150
LR = 0.001
BATCH_SIZE = 512
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
PRINT_EVERY = 10
BACKBONE_WEIGHTS = "weights/ycrcb_backbone.pth"
writer = SummaryWriter(f"logs/{formatted_time}")


def split_input_label_rgb(data):
    grayscale_transform = transforms.Grayscale(num_output_channels=3)
    input = grayscale_transform(data)
    input = (input - RGB_MEAN * 255) / (RGB_STD * 255)

    label = data
    label = label / 255.0
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

    model = ColorNet(None, out_channels=3, optimize_backbone=False).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1,
                total_iters=int(0.02 * EPOCHS),
            ),
            CosineAnnealingLR(
                optimizer,
                T_max=int(0.98 * EPOCHS),
                eta_min=0.05 * LR,
            ),
        ],
        milestones=[int(0.05 * EPOCHS)],
    )
    criterion = nn.L1Loss()
    best_loss = 1e6
    patience = 0

    for epoch in range(EPOCHS):

        model = model.train()
        steps = len(train_loader)
        progress_bar = tqdm(total=steps, desc="Training", position=0)
        for i, data in enumerate(train_loader, 0):
            input, label = split_input_label_rgb(data.to(device))

            output = model(input)
            loss = criterion(output.flatten(1), label.flatten(1))

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
            input, label = split_input_label_rgb(data.to(device))

            with torch.no_grad():
                output = model(input)
            loss = criterion(output, label)
            avg_loss += loss.item()
            if i < 3:
                input = (
                    transforms.Grayscale(num_output_channels=3)(data[0])
                    .permute(1, 2, 0)
                    .numpy()
                )
                output = output[0].cpu().permute(1, 2, 0).numpy()
                label = label[0].cpu().permute(1, 2, 0).numpy()

                output = (output * 255).astype(np.uint8)
                label = (label * 255).astype(np.uint8)

                out = np.concatenate([label, input, output], axis=1)
                writer.add_image(
                    f"Validation/{i}",
                    torch.from_numpy(out).permute(2, 0, 1),
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
