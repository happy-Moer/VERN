# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -6
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]

# Train:
batch_size = 16
cropsize = 128
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 1
shuffle_val = False
val_freq = 10


# Dataset
# TRAIN_PATH = '/home/zhangjiansong/dataset/face/'
# VAL_PATH = '/home/zhangjiansong/dataset/face/'
TRAIN_PATH = ' '
VAL_PATH = ' '
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = 'models/model2/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = 'image_test/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
# IMAGE_PATH = ''
# IMAGE_PATH_cover = IMAGE_PATH
# IMAGE_PATH_secret = IMAGE_PATH
# IMAGE_PATH_steg = IMAGE_PATH
# IMAGE_PATH_secret_rev = IMAGE_PATH


# Load:
suffix = 'model.pt'
tain_next = False
trained_epoch = 0
