
DATA_STATE =  "Raw" #'Transformed' #Preprocessed
TEST_STATE = 'test'
TRAIN_STATE = 'train'
RESULTS_DIR='C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\checkpoints'
BASE_DIR = 'C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\dataset\\lung_image_sets' #'/media/mvisionai/WINDOWS/FED_DATA/latent3'       #
ADENO_DIR = 'C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\dataset\\lung_image_sets\\adeno'  #'/media/mvisionai/WINDOWS/FED_DATA/latent3/ADNI'    #
BENIGH_DIR = 'C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\dataset\\lung_image_sets\\benign' #'/media/mvisionai/WINDOWS/FED_DATA/latent3/NACC'    #
SQUAMOUS_DIR  = 'C:\\Users\\NEW\\Desktop\\FedLung\\FedLung\\dataset\\lung_image_sets\\squamous' #'/media/mvisionai/WINDOWS/FED_DATA/latent3/NACC'    #

latent_dim=64*64
BEN_VRS_ADEN_VRS_SQU = {'benign':0,'adeno':1,'squamous':2}
AUGS = ['noise','flip', 'contrast','blur',"dropout",'invert','average','contrast']
# Number of workers for dataloader
workers = 0
AUGMENT = False
# Batch size during training
batch_size = 5 #5

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
#nz = 64  #100
nz = 4  #100

num_classes = 3
#nz = 100
# Size of feature maps in generator
ngf = 3  #ngf = 64 3
ngf_2 = 8

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.001 #0.0002`

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

encoder_chn = 256


weight_decay = 0.00001