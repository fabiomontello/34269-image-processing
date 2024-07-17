from urllib.request import urlopen

import albumentations as A
import timm
import torch
from albumentations.pytorch.transforms import ToTensorV2
from imagenet1k import load_imagenet_1k
from PIL import Image
from torch import nn
from transforms import Transforms

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 128
DATA_PATH = "./data/imagenet-val/imagenet-val/"
RGB_MEAN = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


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

    # # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(teacher)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    teacher = teacher.eval()
    student = student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    criterion = nn.MSELoss()
    print(teacher)
    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader, 0):
            teacher_data = data.to(device)

            teacher_data = (teacher_data - (RGB_MEAN * 255)) / (RGB_STD * 255)

            with torch.no_grad():
                teacher_output = teacher(teacher_data)

            student_output = student(teacher_data)
            loss = criterion(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
