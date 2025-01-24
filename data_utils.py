import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import utils
import  nibabel as nib
import  numpy as np


def data_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def nifti_load(file):
    data = nib.load(file).get_fdata()
    return data


def npy_load(file):
    data = np.load(file, allow_pickle=True)
    return data

def nifti_slice(data,slice):
    return data[slice,:,:]

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan):
    return np.clip(scan, -1, 2.5)

def normalize(self, data: np.ndarray):
    data_min = np.min(data)
    return (data - data_min) / (np.max(data) - data_min)

def load_by_all(load_state=None,test=False):

    datas = []
    if test == True:
        data_path = os.path.join(utils.BASE_DIR, utils.TEST_STATE)
    else:
        data_path = os.path.join(utils.BASE_DIR, utils.TRAIN_STATE)

    for key, item in load_state.items():
            temp_dir = os.path.join(data_path, key)
            for base_dir, sub_dir, files in os.walk(temp_dir):
                for file in files:
                    current_file_path = os.path.join(base_dir, file)
                    file_path = os.path.join(base_dir, file)
                    datas.append([file_path, int(item),"no_aug"])
                    if utils.AUGMENT and test==False:
                        for aug_type in utils.AUGS:
                            datas.append([file_path, int(item), aug_type])

    return datas




