import torch.utils.data as data
import  nibabel as nib
from utils.data_utils import nifti_load,normalization,clip,nifti_slice, npy_load
import  numpy as np
import torchvision.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from  utils.utils import BASE_DIR
from scipy.ndimage import zoom
import imgaug.augmenters as iaa
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType
from utils import data_utils
import torch
from skimage import io, transform

def augment(img, aug_type):
    augmenter = None
    img =img.astype(np.float32)
    if aug_type == "no_aug":
        return img
    if aug_type == "noise":
        augmenter = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.005 * 255), seed=25, per_channel=True)
        ])
    elif aug_type == "flip":
        augmenter = iaa.Sequential([
            iaa.Fliplr(0.5,seed=25),
            iaa.Flipud(0.5, seed=25)
        ])
    elif aug_type == "contrast":
        augmenter = iaa.Sequential([
            iaa.LinearContrast((0.75, 1.5),seed=25)
        ])
    elif aug_type == "blur":
        augmenter = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 0.3), seed=25)
        ])
    elif aug_type == "dropout":
        augmenter = iaa.Sequential([
            iaa.Dropout(p=(0, 0.2), per_channel=0.5,seed=25)
        ])
    elif aug_type == "invert":
        augmenter = iaa.Sequential([
            iaa.Invert(0.25, per_channel=0.5,seed=25)
        ])
    elif aug_type == "average":
        augmenter = iaa.Sequential([
            iaa.AverageBlur(k=(2,11),seed=25)
        ])
    elif aug_type == "contrast":
        augmenter = iaa.Sequential([
            iaa.GammaContrast((0.5,2.0),seed=25)
        ])

    return augmenter(images=img)


class HospitalDatasetSingle(data.Dataset):
    def __init__(self, dataset, client_index):
        self.dataset = dataset
        self.client_index = client_index
        self.img_transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  Resize((224, 224)),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()])
        #self.img_transform = transforms.Compose([
            #transforms.ToTensor()])

    def __len__(self):
        return len( self.client_index )

    def __getitem__(self, item):
        selected_indice = self.client_index[item]
        image_list = self.dataset[selected_indice]
        image_path = image_list[0]
        label = int(image_list[1])
        image = io.imread(image_path)
        image = np.array([image])
        image = np.squeeze(image)
        #image = np.moveaxis(image, (0, 1, 2), (1, 2, 0))
        image = self.img_transform(image)
        #print(image.shape)
        #image = np.expand_dims(image, axis=0)
        #image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))

        return image, label

