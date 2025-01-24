from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io, transform
import numpy as np
import torchvision.transforms as transforms
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
#best_fedavg_be_vs_ad_sq_model_53_99.73.pt single_model.pt all_single_model.pt
model = models.densenet161(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 3)
restore_model = torch.load(
        os.path.join(os.getcwd(), "checkpoints", "PerFedAvg_be_vs_ad_sq_model.pt"),
        weights_only=True
    )
model.load_state_dict(restore_model['model_state_dict'])
#net_glob.to(device)
#print(model)
#model = resnet50(pretrained=True)
#print(model)
target_layers = [model.features.denseblock4.denselayer16]
#target_layers = [model.layer4[-1]]
#print(target_layers)
#print(models.densenet121(pretrained=False))
#print(target_layers)
#exit(1)

image = io.imread("C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\test_image\\adeno_2.jpeg")
new_image = image/255
new_image = zoom(new_image, (0.292, 0.292, 1))
# print(new_image.shape)
# plt.imshow(new_image)
# plt.show()
image = np.array([image])
image = np.squeeze(image)
#image = np.moveaxis(image, (0, 1, 2), (1, 2, 0))
gray_img = image[:,:,0]
#input_tensor = torch.from_numpy(np.float32(image)).unsqueeze(0) #.unsqueeze(0)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

numpy_transform = transforms.Compose([
    transforms.ToTensor(),
    Resize((224, 224))])

input_tensor = img_transform(image).unsqueeze(0)
numpy_tensor = numpy_transform(image).unsqueeze(0)
#input_tensor = input_tensor.repeat(16, 1, 1, 1)

numpy_img= numpy_tensor.cpu().numpy().squeeze()

#input_tensor = device # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!
#print(image.shape, input_tensor.shape )
#exit(1)
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(2)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
rgb_img =np.round(new_image.astype(np.float32),2)
rgb_img[rgb_img == 1.01] = 1.00
print(rgb_img)
#rgb_img = rgb_img.transpose(1, 2, 0) #np.moveaxis(rgb_img, (0, 1, 2), (1, 2, 0))
#print(rgb_img.shape)
print(np.max(rgb_img))
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.axis('off')
plt.imshow(visualization)
plt.savefig('grad_cam_single_single_adeno_FedPerFedAvg.png', bbox_inches="tight")
#plt.imshow(grayscale_cam)
#plt.imshow(rgb_img)
plt.show()