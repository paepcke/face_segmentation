import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
import training.models
from training.model import createDeepLabv3
from training.trainer import train_model
import training.datahandler as datahandler
import argparse
import os
import torch

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--batchsize", default=4, type=int)
parser.add_argument("--outputConfig", default=3, type=int)
parser.add_argument("--gpuId", default=0, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
outputConfig = args.outputConfig
gpu_id=args.gpuId

# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3(outputConfig)
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}


# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(
    data_dir, imageFolder='images', maskFolder='masks', fraction=0.2, batch_size=batchsize)

trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs, gpu_id=gpu_id)


# Save the trained model
#torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
dst_file_pickled    = os.path.join(bpath, 'whole_model.pt')
dst_file_state_dict = os.path.join(bpath, 'state_dict.pt')
print(f"Saving pickled model to {dst_file_pickled}")
torch.save(trained_model, dst_file_pickled)
print(f"Saving pickled model to {dst_file_state_dict}")
torch.save(trained_model.state_dict(), dst_file_state_dict)
print("Done")

