import modelbit, sys
from typing import *
from _io import BytesIO
from torchvision.models.resnet import ResNet
from torch import Tensor
import requests
import PIL.Image as Image
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import modelbit as mb

resnet50 = modelbit.load_value("data/resnet50.pkl") # ResNet( (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) (relu): ReLU(inplac...
input_batch = modelbit.load_value("data/input_batch.pkl") # tensor([[[[-0.5938, -0.4911, -0.4568, ..., 1.4098, 1.2043, 1.5810], [-0.3198, -0.3027, -0.3369, ..., 1.2728, 1.1529, 1.5125], [-0.1828, -0.1657, -0.2684, ..., 0.8276, 0.9817, 1.2899], ..., [ 1.0673, 1...
labels = modelbit.load_value("data/labels.pkl") # ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', ...

def display_image(inp, predicted_label):    
    # De-normalize the image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = input_batch[0].numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(16,4))
    plt.axis('off')
    plt.imshow(inp)
    plt.title(predicted_label)


# main function
def resnet_inference(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pass the image for preprocessing and reshape to add batch dimension
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    # Predict the class of the image
    with torch.no_grad(): # no_grad ensures the gradients are not calculated in prod
        output = resnet50(input_batch)
    
    _, predicted_idx = torch.max(output, 1)

    pred_img = display_image(img, predicted_label=labels[predicted_idx.item()])
    mb.log_image(pred_img) # show the predicted boxes on the image in modelbit logs

    return { "index": predicted_idx.item(), "label": labels[predicted_idx.item()]}

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(resnet_inference(*(modelbit.parseArg(v) for v in sys.argv[1:])))