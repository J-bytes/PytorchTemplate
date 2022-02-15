#------python import------------------------------------
import warnings
import torch
import tqdm
import copy
from comet_ml import Experiment


#-----local imports---------------------------------------
from training.training import training
from training.dataloaders.galaxy_dataloader import CustomDataset
from models.Unet import Unet


#-------data initialisation-------------------------------
data_path="/mnt/g/data_galaxies/expanded_dataset_v010.h5"
train_dataset=CustomDataset(data_path, method="train", val_size=0.2, test_size=0.7)
val_dataset=copy.copy(train_dataset)
val_dataset.method="val"
training_loader=torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
validation_loader=torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
print("The data has now been loaded successfully into memory")
#-----------model initialisation------------------------------
if torch.cuda.is_available() :
    device="cuda:0"
else :
    device="cpu"
    warnings.warn("No gpu is available for the computation")

model=Unet(depth=2,channels=[1,2,3]).to(device)
optimizer=torch.optim.AdamW(model.parameters())
criterion=torch.nn.KLDivLoss() # to replace
print("The model has now been successfully loaded into memory")

#---comet logger initialisation
# Create an experiment with your api key
#experiment = Experiment(
#    api_key="HLrroRFl9Ay2kurjtwuq6Kmq9",
#    project_name="ift6759",
#    workspace="bariljeanfrancois",
#)
print("Starting training now")
training(model,optimizer,criterion,training_loader,validation_loader,device,verbose=False)