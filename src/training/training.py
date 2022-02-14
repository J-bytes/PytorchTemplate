import warnings

import torch
import tqdm
# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
#experiment = Experiment(
#    api_key="HLrroRFl9Ay2kurjtwuq6Kmq9",
#    project_name="ift6759",
#    workspace="bariljeanfrancois",
#)

# Run your code and go to /

# local imports
from src.models.Unet import Unet


# Create datasets for training & validation, download if necessary
# import hub
# training_set =hub.load("hub://activeloop/coco-train")
# validation_set =hub.load("hub://activeloop/coco-val")

# Create data loaders for our datasets; shuffle for training, not for validation
# training_loader =  training_set.pytorch(num_workers=0, batch_size=4, shuffle=False)#torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
# validation_loader = validation_set.pytorch(num_workers=0, batch_size=4, shuffle=False)#torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=4,pin_memory=True)
from src.training.dataloaders.brain_tumour_dataloader import BrainSegmentationDataset
dataset1=BrainSegmentationDataset("data/medical_data/Brain_tumours/imagesTr")
dataset2=BrainSegmentationDataset("data/medical_data/Brain_tumours/imagesTs",subset="valid")
training_loader=torch.utils.data.DataLoader(dataset1, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
validation_loader=torch.utils.data.DataLoader(dataset2, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)

# Report split sizes
#print('Training set has {} instances'.format(len(training_set)))
#print('Validation set has {} instances'.format(len(validation_set)))

# global variables
if torch.cuda.is_available() :
    device="cuda:0"
else :
    device="cpu"
    warnings.warn("No gpu is available for the computation")

model=Unet(depth=3,channels=[1,2,3]).to(device)
optimizer=torch.optim.AdamW(model.parameters())
criterion=torch.nn.BCELoss() # to replace

# training loop

def training_loop(model,loader,optimizer,criterion) :
    running_loss=0
    for data in tqdm.tqdm(loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data.tensors.images,data.tensors
        inputs,labels=inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.detach()
    return running_loss



def validation_loop(model,loader,criterion):
    running_loss=0
    with torch.no_grad() :
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss+=loss.detach()
    return running_loss

previous_loss=1000
current_loss=0
epoch,epoch_max=0,150
while (current_loss-previous_loss)<0 and epoch<epoch_max:  # loop over the dataset multiple times

    running_loss = 0.0
    train_loss=training_loop(model,training_loader,optimizer,criterion)
    val_loss=validation_loop(model,validation_loader,criterion)

    #other evaluation metrics to display :
    #f1_loss, etc

    #log the results :


    #save the model after XX iterations :


    #Finishing the loop
    epoch+=1
print('Finished Training')