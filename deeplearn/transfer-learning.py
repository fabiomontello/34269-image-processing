import datetime
from urllib.request import urlopen

import albumentations as A
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from imagenet1kfull import load_imagenet_1k
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transforms import Transforms

# Get the current date and time
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 128
DATA_PATH = "/home/fabmo/works/mod-vit/data/imagenet-1k/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
PRINT_EVERY = 10
writer = SummaryWriter(f"logs/{formatted_time}")


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            "Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape)
        )

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack((y, cb, cr), -3)


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Assuming the model is a nn.Sequential, remove the last layer (head)
        # This is a generic approach; the actual implementation may vary based on the model architecture
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RGB_MEAN = RGB_MEAN.to(device)
    RGB_STD = RGB_STD.to(device)

    train_transform = Transforms(train=True, image_size=224)
    test_transform = Transforms(train=False, image_size=224)

    # Load the ImageNet dataset
    train_loader, val_loader, test_loader = load_imagenet_1k(
        data_path=DATA_PATH,
        transform_train=train_transform,
        transform_test=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
    teacher = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    teacher = FeatureExtractor(teacher).to(device)

    student = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,  # remove classifier nn.Linear
    ).to(device)
    student = FeatureExtractor(student).to(device)

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(teacher)
    # transforms = timm.data.create_transform(**data_config, is_training=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_loss = 1000
    patience = 0
    for epoch in range(EPOCHS):

        teacher = teacher.eval()
        student = student.train()
        steps = len(train_loader)
        progress_bar = tqdm(total=steps, desc="Training", position=0)
        for i, data in enumerate(train_loader, 0):
            teacher_data = data[0].to(device)

            teacher_data = (teacher_data - (RGB_MEAN * 255)) / (RGB_STD * 255)

            with torch.no_grad():
                teacher_output = teacher(teacher_data)

            student_data = rgb_to_ycbcr(teacher_data)
            student_output = student(student_data)
            loss = criterion(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % PRINT_EVERY == 0:
                writer.add_scalar(
                    "Loss/train",
                    loss,
                    epoch * len(train_loader) + i,
                )
                progress_bar.set_postfix({"Loss": f"{(loss):.4f}"})
                progress_bar.update(PRINT_EVERY)

        # evaluate the student model
        student = student.eval()
        avg_loss = 0
        for i, data in enumerate(tqdm(val_loader), 0):
            teacher_data = data[0].to(device)
            teacher_data = (teacher_data - (RGB_MEAN * 255)) / (RGB_STD * 255)
            student_data = rgb_to_ycbcr(teacher_data)
            with torch.no_grad():
                teacher_output = teacher(teacher_data)
                student_output = student(student_data)
            loss = criterion(student_output, teacher_output)
            avg_loss += loss.item()
        avg_loss /= len(val_loader)
        print(f"Epoch: {epoch}, Loss: {avg_loss}")
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
