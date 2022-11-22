#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""

import logging
import os
import pathlib

import numpy as np
import torch
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data.sampler import SequentialSampler
import wandb
from CheXpert2.custom_utils import convert
from CheXpert2.results_visualization import plot_polar_chart
from CheXpert2.training.training import training_loop,validation_loop
from CheXpert2.dataloaders.CXRLoader import CXRLoader
import torch.distributed as dist
import tqdm
from CheXpert2 import names
import torch
import numpy as np
import pandas as pd
from DeepLearningArchitectures.Metrics import Metrics  # sklearn f**ks my debug


class Experiment:
    def __init__(self, directory : str, names: [str], tag : str = None, config=None, epoch_max : int=50, patience : int=5,verbose : int=1):
        """
        Initalize the experiment with some prerequired information.

        Parameters :
        -------------------------------------------------------------

        directory : Str
            directory on which the images are stored
        names : List
            List of strings of the classes names
        tag : Str
            A string used to tag the run on Wandb
        config : Dict
            Dict of the config for the model as given by Parser.py
        epoch_max : Int
            The Number of epoch the experiment can run
        patience : Int
            Amount of patience of the experiment. Default is 5
        verbose : Int
            Integer between 0 & 5. Default is 1
        """
        self.names = names
        self.weight_dir = "models_weights/" + directory

        self.summary = {}
        self.metrics_results = {}
        self.pbar = tqdm.tqdm(total=epoch_max, position=0)
        self.best_loss = np.inf
        self.keep_training = True
        self.epoch = 0
        self.epoch_max = epoch_max
        self.max_patience = patience
        self.patience = patience
        self.rank = int(os.environ['LOCAL_RANK']) if torch.distributed.is_initialized() else 0
        self.names = names

        # create directory if doesnt existe
        path = pathlib.Path(self.weight_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            wandb.init(project="Chestxray", entity="ccsmtl2", config=config,tags=tag)

        self.verbose=verbose


    def next_epoch(self, val_loss):
        if self.rank == 0 :
            if val_loss < self.best_loss or self.epoch == 0:
                self.best_loss = val_loss
                self.log_metric("best_loss", self.best_loss, epoch=self.epoch)
                self.patience = self.max_patience
                self.summarize()
                self.save_weights()
            else:
                self.patience -= 1

                logging.info(f"patience has been reduced by 1, now at {self.patience}")
                logging.info(f"training loss : {self.metrics_results['training_loss']}")
                logging.info(f"validation loss : {val_loss}")
            self.pbar.update(1)
            logging.info(pd.DataFrame(self.metrics_results, columns=list(self.metrics_results.keys())))
        self.epoch += 1
        if self.patience == 0 or self.epoch == self.epoch_max:
            self.keep_training = False




    def log_metric(self, metric_name : str, value, epoch=None):
        if self.rank == 0 :
            if epoch is not None:
                wandb.log({metric_name: value, "epoch": epoch})
            else:
                wandb.log({metric_name: value})
            self.metrics_results[metric_name] = value

    def log_metrics(self, metrics : dict, epoch=None) :
        if self.rank == 0 :
            metrics["epoch"] = epoch
            wandb.log(metrics)
            self.metrics_results = self.metrics_results | metrics

    def save_weights(self):
        if self.rank == 0 and os.environ["DEBUG"] == "False" :
            if dist.is_initialized() :
                torch.save(self.model.module.state_dict(), f"{self.weight_dir}/{self.model.module.backbone._get_name()}.pt")
                wandb.save(f"{self.weight_dir}/{self.model.module.backbone._get_name()}.pt")
            else :
                torch.save(self.model.state_dict(), f"{self.weight_dir}/{self.model.backbone._get_name()}.pt")
                wandb.save(f"{self.weight_dir}/{self.model.backbone._get_name()}.pt")

    def summarize(self):
        self.summary = self.metrics_results


    def watch(self, model):
        if self.rank == 0 :
            wandb.watch(model)

    def end(self, results):

        for key,value in self.summary.items():
            wandb.run.summary[key] = value
        if self.rank == 0 :
            # 1) confusion matrix

            self.log_metric(
                "conf_mat",
                wandb.sklearn.plot_confusion_matrix(
                    convert(results[0]),
                    convert(results[1]),
                    self.names,
                ),
                epoch=None)
            plot_polar_chart(self.summary)

    def compile(self,model,optimizer : str or None,criterion : str or None ,train_datasets : [str],val_datasets : [str],config,device) :
        """
            Compile the experiment before training

            This function simply prepare all parameters before training the experiment

            Parameters
            ----------
            model : Subclass of torch.nn.Module
                A pytorch model
            optimizer :  String, must be a subclass of torch.optim
                -Adam
                -AdamW
                -SGD
                -etc.

            criterion : String , must be a subclass of torch.nn
                - BCEWithLogitsLoss
                - BCELoss
                - MSELoss
                - etc.

            train_datasets : List of string
            Datasets from which to select the training data. Choices include :
                - ChexPert
                - ChexNet
                - CIUSSS
                - etc

            val_datasets : List of string
            Datasets from which to select the validation data. Choices include :
                - ChexPert
                - ChexNet
                - CIUSSS
                - etc
            config : dict
                The config returned by the parser file

            device : str
                The device to use for training.
                    - "cpu"
                    - "cuda:$x" with x being the GPU number (0 by default)

        """
        self.model=model.to(device,dtype=torch.float)
        self.device = device
        self.watch(self.model)
        self.config = config
        self.num_classes = len(names)
        assert optimizer in dir(torch.optim)+[None]
        assert criterion in dir(torch.nn)+[None]


        img_dir=os.environ["img_dir"]

        train_dataset =CXRLoader(
            split="Train",
            img_dir=img_dir,
            img_size=config["img_size"],
            prob=config["augment_prob"],
            intensity=config["augment_intensity"],
            label_smoothing=config["label_smoothing"],
            channels=config["channels"],
            use_frontal=config["use_frontal"],
            datasets=train_datasets
        )
        val_dataset=CXRLoader(
                split="Valid",
                img_dir=img_dir,
                img_size=config["img_size"],
                prob=[0,0,0,0,0,0,0],
                intensity=0,
                label_smoothing=0,
                channels=config["channels"],
                use_frontal=config["use_frontal"],
                datasets=val_datasets
        )
        num_positives = train_dataset.count
        num_negatives = len(train_dataset) - num_positives
        #thresholds    =  num_positives / num_negatives

        thresholds = np.zeros((self.num_classes)) + 0.5
        metric = Metrics(num_classes=self.num_classes, names=names, threshold=thresholds)
        self.metrics = metric.metrics()
        threshold_log={}
        threshold_log["thresholds"] = {name: threshold for name, threshold in zip(self.names, thresholds.tolist())}
        self.log_metrics(threshold_log)
        if logging :
            logging.debug(f"Loaded {len(train_dataset)} exams for training")
            logging.debug(f"Loaded {len(val_dataset)} exams for validation")
        if os.environ["DEBUG"] == "False":
            num_samples = 10_000
        else:
            num_samples = 10


        if train_dataset.weights is not None:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weights,
                                                                     num_samples=min(num_samples, len(train_dataset)))
        else:
            sampler = torch.utils.data.SubsetRandomSampler(
                list(range(len(train_dataset)))[0:min(num_samples, len(train_dataset))], generator=None)

        if dist.is_initialized():
            sampler = torch.utils.data.DistributedSampler(SequentialSampler(sampler))

        self.training_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_worker"],
            pin_memory=True,
            sampler=sampler,

        )
        self.validation_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_worker"],
            pin_memory=True,
            shuffle=False,
        )
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            betas=(self.config["beta1"], self.config["beta2"]),
        ) if optimizer else None


        self.criterion = getattr(torch.nn, criterion) if criterion else None


    def train(self,**kwargs):
        """
        Run the training for the compiled experiment

        This function simply prepare all parameters before training the experiment

        Parameters
        ----------
        **kwargs : Override default methods in compile

        """
        for key, value in kwargs.items():
            setattr(self,key,value)


        self.epoch = 0
        results = None

        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["autocast"])
        val_loss = torch.inf
        n, m = len(self.training_loader), len(self.validation_loader)



        position = self.device + 1 if type(self.device) == int else 1
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10,T_mult=1)
        #scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config["lr"], steps_per_epoch=len(self.training_loader), epochs=self.epoch_max)

        with logging_redirect_tqdm():

            while self.keep_training:  # loop over the dataset multiple times
                metrics_results = {}
                if dist.is_initialized():
                    self.training_loader.sampler.set_epoch(self.epoch)

                train_loss = training_loop(
                    self.model,
                    tqdm.tqdm(self.training_loader, leave=False, position=position),
                    self.optimizer,
                    self.criterion,
                    self.device,
                    scaler,
                    self.config["clip_norm"],
                    self.config["autocast"],
                    scheduler,
                    epoch=self.epoch
                )
                if self.rank == 0:
                    val_loss, results = validation_loop(
                        self.model, tqdm.tqdm(self.validation_loader, position=position, leave=False), self.criterion, self.device,self.config["autocast"]
                    )
                    logging.debug(f"mean output : {torch.mean(results[1])}")


                    if self.metrics:
                        for key,metric in self.metrics.items():
                            pred = results[1].numpy()
                            true = results[0].numpy().round(0)

                            metric_result = metric(true, pred)
                            metrics_results[key] = metric_result

                        self.log_metrics(metrics_results, epoch=self.epoch)
                        self.log_metric("training_loss", train_loss.cpu() / n, epoch=self.epoch)
                        self.log_metric("validation_loss", val_loss.cpu() / m, epoch=self.epoch)

                    # Finishing the loop

                self.next_epoch(val_loss.cpu() / m)
                # if not dist.is_initialized() and self.epoch % 5 == 0:
                #     set_parameter_requires_grad(model, 1 + self.epoch // 2)
                if self.epoch == self.epoch_max:
                    self.keep_training = False
            if logging :
                logging.info("Finished Training")
            return results








if __name__ == "__main__":
    experiment = Experiment(dir="/debug", names=["1", "2", "3"])
    os.environ["DEBUG"] = "True"

    experiment.log_metric("auc", {"banana": 0})

    experiment.log_metrics({"apple": 3})

    experiment.next_epoch(3, None)

    results = [torch.randint(0, 2, size=(10, 13)), torch.rand(size=(10, 13))]
    experiment.end(results)
