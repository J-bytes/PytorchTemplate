from torchvision import transforms
import os
import torch
import pathlib
import sklearn
import numpy as np
from  sklearn.metrics import top_k_accuracy_score
import wandb
#-----------------------------------------------------------------------------------
class Experiment() :
    def __init__(self,directory,is_wandb=False):
        self.is_wandb=is_wandb
        self.directory="log/"+directory
        self.weight_dir="models/models_weights/"+directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        path=pathlib.Path(self.weight_dir)
        path.mkdir(parents=True,exist_ok=True)

        root,dir,files = list(os.walk(self.directory))[0]
        for f in files:
            os.remove(root+"/"+f)



    def log_metric(self,metric_name,value,epoch):

        f=open(f"{self.directory}/{metric_name}.txt","a")
        if type(value)==list :
            f.write("\n".join(str(item) for item in value))
        else :
            f.write(f"{epoch} , {str(value)}")

        if self.is_wandb :
            wandb.log({metric_name: value})
    def save_weights(self,model):

        torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")

#-----------------------------------------------------------------------------------
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
#-----------------------------------------------------------------------------------
class preprocessing() :
    def __init__(self,img_size,other=None):
        self.img_size=img_size
        self.added_transform=other

    def preprocessing(self):
        temp=[
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ]
        if self.added_transform :
            temp.append(self.added_transform)
        preprocess = transforms.Compose(temp)
        return preprocess
#-----------------------------------------------------------------------------------
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

#-----------------------------------------------------------------------------------
num_classes = 14 #+empty



class metrics :
    def __init__(self,num_classes):
        self.num_classes=num_classes


    def top1(self,true, pred):
        true = np.argmax(true, axis=1)
        # labels=np.unique(true)
        labels = np.arange(0, self.num_classes)

        return top_k_accuracy_score(true, pred, k=1, labels=labels)

    def top5(self,true, pred):
        true = np.argmax(true, axis=1)
        labels = np.arange(0, self.num_classes)

        return top_k_accuracy_score(true, pred, k=5, labels=labels)

    def f1(self,true, pred):
        true = np.argmax(true, axis=1)
        pred = np.argmax(pred, axis=1)

        return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??

    def precision(self,true, pred):
        true = np.argmax(true, axis=1)
        pred = np.argmax(pred, axis=1)
        return sklearn.metrics.precision_score(true, pred, average='macro')

    def recall(self,true, pred):
        true = np.argmax(true, axis=1)
        pred = np.argmax(pred, axis=1)
        return sklearn.metrics.recall_score(true, pred, average='macro')

    def metrics(self):
        dict={
            "f1" : self.f1,
            "top1" : self.top1,
            "top5" : self.top5,
            "recall" : self.recall,
            "precision" : self.precision
        }
        return dict