import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import models, transforms, datasets

import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
import shutil

torch.manual_seed(0)

train_dir = os.path.join("data", "train_steve")
val_dir = os.path.join("data", "val_steve")

# Resize the samples and transform them into tensors
data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

train_dataset = datasets.ImageFolder(train_dir, data_transforms)
val_dataset = datasets.ImageFolder(val_dir, data_transforms)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
print("The classes are: ", class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataloaders initialization
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

model_ft = models.resnet50(pretrained=False)

# Fit the last layer for our specific task
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load('trained_model.pt'))

model_ft.eval()

sm = nn.Softmax()
threshold = 0.7

w = 0
u = 0
wrong = []
unsure = []
for dataset in [train_dataset, val_dataset]:
    for (input, label), img in tqdm(zip(dataset, dataset.imgs)):
        input = input.to(device)
        input = input.unsqueeze(0)
        outputs = model_ft(input)
        prob, pred = torch.max(sm(outputs), 1)

        if pred.item() != label:
            w += 1
            wrong.append((img[0], pred.item()))
        elif prob < threshold:
            u += 1
            unsure.append((img[0], pred.item()))


print(f'The model was wrong about {w} images. They are:')
# print(wrong)

print(f'The model was unsure about {u} images. They are:')
# print(unsure)


for lst, name in zip([wrong, unsure], ['wrong', 'unsure']):
    if os.path.exists('data/' + name):
        shutil.rmtree('data/' + name)
    os.makedirs('data/' + name)
    for img in lst:
        file_name = img[0].split('/')
        shutil.copyfile(img[0], 'data/' + name + '/' + file_name[-1])



