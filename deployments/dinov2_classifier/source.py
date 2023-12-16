import modelbit, sys
from typing import *
from _io import BytesIO
from dinov2.hub.classifiers import _LinearClassifierWrapper
import requests
import PIL.Image as Image
import torchvision.transforms as T
import torch

device = modelbit.load_value("data/device.pkl") # cuda
dinov2_vitg14_reg_lc = modelbit.load_value("data/dinov2_vitg14_reg_lc.pkl") # _LinearClassifierWrapper( (backbone): DinoVisionTransformer( (patch_embed): PatchEmbed( (proj): Conv2d(3, 768, kernel_size=(14, 14), stride=(14, 14)) (norm): Identity() ) (blocks): ModuleList( (0-11):...
imagenet_classes = modelbit.load_value("data/imagenet_classes.pkl") # ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', ...

# main function
def dinov2_classifier(img_url):
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))

    # Preprocess the image
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = transform(image)

    # Move the image to the GPU if available
    image = image.to(device)

    # Extract the features
    with torch.no_grad():
        features = dinov2_vitg14_reg_lc(image.unsqueeze(0))

    # Print the features
    return {'index': features.argmax(-1).item(),
            'label': imagenet_classes[features.argmax(-1).item()]
    }

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(dinov2_classifier(*(modelbit.parseArg(v) for v in sys.argv[1:])))