
import torch
import tqdm








# Create datasets for training & validation, download if necessary
# import hub
# training_set =hub.load("hub://activeloop/coco-train")
# validation_set =hub.load("hub://activeloop/coco-val")

# Create data loaders for our datasets; shuffle for training, not for validation
# from src.training.dataloaders.brain_tumour_dataloader import BrainSegmentationDataset
# dataset1=BrainSegmentationDataset("data/medical_data/BraTS2021_Training_Data/training",subset="train")
# dataset2=BrainSegmentationDataset("data/medical_data/BraTS2021_Training_Data/validation",subset="validation")
# training_loader=torch.utils.data.DataLoader(dataset1, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)
# validation_loader=torch.utils.data.DataLoader(dataset2, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)


# training loop

def training_loop(model,loader,optimizer,criterion,device,verbose,epoch,metrics) :
    running_loss=0
    i=0
    metrics_results = {}
    if metrics :
        for key in metrics:
            metrics_results[key] = 0
    for inputs,labels in loader:
        # get the inputs; data is a list of [inputs, labels]

        inputs,labels=inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.detach()

        if verbose and i % 20 == 0:
            print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")

        if metrics :
            for key in metrics:
                metrics_results[key]+= metrics[key](outputs,labels)/len(inputs)

        #ending loop
        del inputs,labels,loss,outputs #garbage management sometimes fails with cuda
        i+=1
    return running_loss,metrics_results



def validation_loop(model,loader,criterion,device,verbose,epoch,metrics):
    running_loss=0
    i=0

    metrics_results={}
    if metrics :
        for key in metrics :
            metrics_results[key]=0
    with torch.no_grad() :
        for inputs,labels in loader:
            # get the inputs; data is a list of [inputs, labels]

            inputs,labels=inputs.to(device),labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss+=loss.detach()

            if verbose and i%20==0 :
                print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")

            if metrics :
                for key in metrics:
                    metrics_results[key] += metrics[key](outputs, labels) / len(inputs)
            #ending loop
            del inputs,labels,outputs,loss #garbage management sometimes fails with cuda
            i+=1
    return running_loss,metrics_results



def training(model,optimizer,criterion,training_loader,validation_loader,device="cpu",metrics=None,verbose=False,experiment=None) :
    previous_loss=1000
    current_loss=0
    epoch,epoch_max=0,150

    if not verbose :
        training_loader=tqdm.tqdm(training_loader)
        validation_loader=tqdm.tqdm(validation_loader)

    train_loss_list=[]
    val_loss_list=[]
    while (current_loss-previous_loss)<0 and epoch<epoch_max:  # loop over the dataset multiple times


        train_loss,metrics_results=training_loop(model,training_loader,optimizer,criterion,device,verbose,epoch,metrics)
        val_loss,metrics_results=validation_loop(model,validation_loader,criterion,device,verbose,epoch.metrics)

        #other evaluation metrics to display :


        #log the results :
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if experiment :
            experiment.log_metric("training_loss",train_loss,epoch=epoch)
            experiment.log_metric("validation_loss", val_loss,epoch=epoch)
            for key in metrics_results :
                experiment.log_metric(key,metrics_results[key],epoch=epoch)
            #et

        #save the model after XX iterations :
        if epoch%20==0 :
            torch.save(model.state_dict(),"models/models_weights/")

        #Finishing the loop
        epoch+=1
    print('Finished Training')