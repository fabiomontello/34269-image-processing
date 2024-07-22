import datetime
from urllib.request import urlopen

import albumentations as A
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from imagenet1k import load_imagenet_1k
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transforms import TransformsPretrain
from utils import FeatureExtractor, rgb_to_ycbcr

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

EPOCHS = 500
LR = 0.001
BATCH_SIZE = 172
DATA_PATH = "/home/fabmo/works/34269-image-processing/data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
YCRCB_MEAN = torch.Tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
YCRCB_STD = torch.Tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

PRINT_EVERY = 10
writer = SummaryWriter(f"logs/{formatted_time}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RGB_MEAN = RGB_MEAN.to(device)
    RGB_STD = RGB_STD.to(device)
    YCRCB_MEAN = YCRCB_MEAN.to(device)
    YCRCB_STD = YCRCB_STD.to(device)

    train_transform = TransformsPretrain(train=True, image_size=224)
    test_transform = TransformsPretrain(train=False, image_size=224)

    # Load the ImageNet dataset
    train_loader, val_loader, test_loader = load_imagenet_1k(
        data_path=DATA_PATH,
        transform_train=train_transform,
        transform_test=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
    teacher = timm.create_model(
        "vit_base_patch16_224.mae",
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    teacher = FeatureExtractor(teacher).to(device)

    student = timm.create_model(
        "vit_base_patch16_224.mae",
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    ).to(device)
    student = FeatureExtractor(student).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
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
    best_loss = 1000
    patience = 0

    for epoch in range(EPOCHS):

        teacher = teacher.eval()
        student = student.train()
        steps = len(train_loader)
        progress_bar = tqdm(total=steps, desc="Training", position=0)
        for i, data in enumerate(train_loader, 0):
            teacher_data = data.to(device)

            teacher_data = (teacher_data - (RGB_MEAN * 255)) / (RGB_STD * 255)

            with torch.no_grad():
                teacher_output = teacher(teacher_data)

            student_data = rgb_to_ycbcr(teacher_data)
            student_data = (student_data - YCRCB_MEAN) / YCRCB_STD
            student_output = student(student_data)
            loss = criterion(student_output, teacher_output)

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
        student = student.eval()
        avg_loss = 0
        for i, data in enumerate(tqdm(val_loader), 0):
            teacher_data = data.to(device)
            teacher_data = (teacher_data - (RGB_MEAN * 255)) / (RGB_STD * 255)
            student_data = rgb_to_ycbcr(teacher_data)
            student_data = (student_data - YCRCB_MEAN) / YCRCB_STD
            with torch.no_grad():
                teacher_output = teacher(teacher_data)
                student_output = student(student_data)
            loss = criterion(student_output, teacher_output)
            avg_loss += loss.item()
        avg_loss /= len(val_loader)
        print(f"Epoch: {epoch}, Loss: {avg_loss}")
        scheduler.step()

        writer.add_scalar("Loss/val", avg_loss, epoch)
        patience += 1
        # Save the model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(student.state_dict(), f"logs/{formatted_time}/student_best.pth")

        if patience > int(EPOCHS * 0.1):
            print("Early stopping")
            break
