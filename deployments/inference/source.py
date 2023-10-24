import modelbit, sys
from typing import *
from _io import BytesIO
from torchvision.models.resnet import ResNet
import requests
import PIL.Image as Image
import torchvision.transforms as transforms
import torch

resnet50 = modelbit.load_value("data/resnet50.pkl") # ResNet( (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) (relu): ReLU(inplac...
labels = modelbit.load_value("data/labels.pkl") # ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', ...

# main function
def inference(img_url):
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
    with torch.no_grad():
        output = resnet50(input_batch)

    _, predicted_idx = torch.max(output, 1)
    return { "index": predicted_idx.item(), "label": labels[predicted_idx.item()]}

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(inference(*(modelbit.parseArg(v) for v in sys.argv[1:])))