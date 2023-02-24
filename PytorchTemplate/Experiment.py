#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""
import wandb
import logging
import sys
import os
import pathlib
import inspect
import warnings
from tqdm.contrib.logging import logging_redirect_tqdm

import torch.distributed as dist
import torch_optimizer

from PytorchTemplate.Metrics import metrics
import tqdm
import timm
from torch.optim.swa_utils import AveragedModel,SWALR
import torch
import numpy as np
import pandas as pd



class Experiment:
    def __init__(self,  names: [str],config : dict,verbose=1):
        """
        Initalize the experiment with some prerequired information.

        Parameters :
        -------------------------------------------------------------


        names : List
            List of strings of the classes names

        config : Dict
            Dict of the config for the model as given by Parser.py

        verbose : Int
            Integer between 0 & 5. Default is 5
        """




        self.names = names
        self.num_classes = len(names)
        self.weight_dir = "models_weights/"
        self.metrics = metrics
        self.config = config
        self.summary = {}
        self.metrics_results = {}
        self.pbar = tqdm.tqdm(total=config["epoch"], position=0)
        self.best_loss = np.inf
        self.keep_training = True
        self.epoch = 0
        self.epoch_max = config["epoch"]
        self.max_patience = config["patience"]
        self.patience = config["patience"]
        self.rank = int(os.environ['LOCAL_RANK']) if torch.distributed.is_initialized() else 0
        self.names = names
        self.autocast = config["autocast"]
        self.clip_norm = config["clip_norm"]
        self.use_features = False

        #  ----------create directory if doesnt existe ------------------------
        path = pathlib.Path(self.weight_dir)
        path.mkdir(parents=True, exist_ok=True)



        #  ---------- initialize the logger  ---------------------------------
        logging.basicConfig(filename='PytorchTemplate/PytorchTemplate.log', level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(verbose)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(verbose*10)

        root.addHandler(handler)

        # 1) set up debug env variable

        if config["debug"]:
            os.environ["WANDB_MODE"] = "offline"

        # ---------- Device Selection ----------------------------------------
        torch.set_num_threads(max(config["num_worker"], 1))
        if torch.cuda.is_available():
            if dist.is_initialized():
                rank = dist.get_rank()
                device = rank % torch.cuda.device_count()
            else:
                device = config["device"]

        else:
            device = "cpu"
            warnings.warn("No gpu is available for the computation")

        self.device = device

        # --------- instantiate experiment tracker -----------------------
        if config["wandb"]:
            project = ""
            entity = ""
            if project == "" or entity == "":
                raise ValueError("Please specify a project name and entity name for wandb")
            run = wandb.init(project=project, entity=entity, config=config)
        else:
            run = None

        self.tracker = run

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
                if self.rank==0 :

                    logging.info(pd.DataFrame(self.metrics_results,index=[0]))
            self.epoch += 1
            if self.patience == 0 or self.epoch == self.epoch_max:
                self.keep_training = False




    def log_metric(self, metric_name : str, value, epoch=None):
        if self.rank == 0:
            if self.tracker is not None:
                if epoch is not None :
                    self.tracker.log({metric_name: value, "epoch": epoch})
                else:
                    self.tracker.log({metric_name: value})
            self.metrics_results[metric_name] = value

    def log_metrics(self, metrics : dict, epoch=None) :
        if self.rank == 0 :
            metrics["epoch"] = epoch
            if self.tracker is not None:
                self.tracker.log(metrics)
            self.metrics_results = self.metrics_results | metrics

    def save_weights(self):
        if self.rank == 0 :
            if dist.is_initialized() :
                torch.save(self.model.module.state_dict(), f"{self.weight_dir}/{self.model.module.name}.pt")
                torch.save(self.optimizer.module.state_dict(), f"{self.weight_dir}/optimizer_{self.model.module.name}.pt")
                if self.tracker is not None:
                    self.tracker.save(f"{self.weight_dir}/{self.model.module.name}.pt")
            else :
                torch.save(self.model.state_dict(), f"{self.weight_dir}/{self.model.name}.pt")
                torch.save(self.optimizer.state_dict(), f"{self.weight_dir}/optimizer_{self.model.name}.pt")
                if self.tracker is not None:
                    self.tracker.save(f"{self.weight_dir}/{self.model.name}.pt")


    def summarize(self):
        self.summary = self.metrics_results


    def watch(self, model):
        if self.rank == 0 and self.tracker is not None:
            self.tracker.watch(model)

    def end(self):
        if self.tracker is not None:
            for key,value in self.summary.items():
                self.tracker.run.summary[key] = value
            self.tracker.finish()

    def compile(self,model_name,train_loader,val_loader,optimizer : str or None,criterion : str or None ,final_activation) :
        """
            Compile the experiment before variation

            This function simply prepare all parameters before variation the experiment

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

            final_activation : String, must be a subclass of torch.nn
                - Sigmoid
                - Softmax
                - etc.


        """

        # -----------model initialisation------------------------------

        model = timm.create_model(model_name, num_classes=self.num_classes, pretrained=True,
                                  drop_rate=self.config["drop_rate"]).to(self.device)
        name = model.default_cfg["architecture"]
        model = AveragedModel(model)
        setattr(model, "name", name)
        self.model = model


        # send model to gpu

        print("The model has now been successfully loaded into memory")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model=model.to(self.device,dtype=torch.float)

        self.watch(self.model)


        self.num_classes = len(self.names)

        assert optimizer in dir(torch.optim)+[None]
        assert criterion in dir(torch.nn)+[None]





        if optimizer in dir(torch.optim):
            optimizer = getattr(torch.optim, optimizer)


        elif optimizer in dir(torch_optimizer) :
            optimizer = getattr(torch.optim, optimizer)

        else :
            raise NotImplementedError("The optimizer is not implemented yet")

        signature = inspect.signature(optimizer.__init__)
        optimizer_params = {key : self.config[key] for key in list(signature.parameters.keys())[2::] if key in self.config }


        self.optimizer = optimizer(
            self.model.parameters(),
            **optimizer_params
        )
        logging.debug(f"Optimizer : {self.optimizer}")

        if criterion in dir(torch.nn):
            self.criterion = getattr(torch.nn, criterion)()
        else :
            raise NotImplementedError("The criterion is not implemented yet")


        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config["lr"], steps_per_epoch=len(self.train_loader), epochs=self.epoch_max)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config["lr"])
        self.swa_start = self.config["swa_start"]

        if final_activation.lower() == "sigmoid" :
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation.lower() == "softmax" :
            self.final_activation = torch.nn.Softmax(dim=1)
        else :
            raise NotImplementedError(f"The activation asked has not been implemented for {final_activation}")


    def train(self,**kwargs):
        """
        Run the variation for the compiled experiment

        This function simply prepare all parameters before variation the experiment

        Parameters
        ----------
        **kwargs : Override default methods in compile

        """
        for key, value in kwargs.items():
            assert key in dir(self), f"You are trying to override {key}. This is not an attribute of the class Experiment"
            setattr(self,key,value)



        if os.path.exists(f"{self.weight_dir}/{self.model.name}.pt"):
            print(f"Loading pretrained weights from {self.weight_dir}/{self.model.name}.pt")
            self.model.load_state_dict(torch.load(f"{self.weight_dir}/{self.model.name}.pt"))
            self.optimizer.load_state_dict(torch.load(f"{self.weight_dir}/optimizer_{self.model.name}.pt"))

        # Creates a GradScaler once at the beginning of variation.
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["autocast"])

        n, m = len(self.train_loader), len(self.val_loader)




        val_loss, results = self.validation_loop()
        self.best_loss = val_loss.cpu().item() / m
        logging.info(f"Starting training with validation loss : {self.best_loss}")
        with logging_redirect_tqdm():

            while self.keep_training:  # loop over the dataset multiple times
                metrics_results = {}
                if dist.is_initialized():
                    self.train_loader.sampler.set_epoch(self.epoch)

                train_loss = self.training_loop()
                if self.rank == 0:
                    val_loss, results = self.validation_loop()



                    if self.metrics:
                        for key,metric in self.metrics.items():
                            pred = self.final_activation(results[1]).numpy()
                            true = results[0].numpy().round(0)

                            metric_result = metric(true, pred)
                            metrics_results[key] = metric_result

                        self.log_metrics(metrics_results, epoch=self.epoch)
                        self.log_metric("training_loss", train_loss.cpu().item() / n, epoch=self.epoch)
                        self.log_metric("validation_loss", val_loss.cpu().item() / m, epoch=self.epoch)


                    # Finishing the loop

                self.next_epoch(val_loss.cpu().item() / m)
                # if not dist.is_initialized() and self.epoch % 5 == 0:
                #     set_parameter_requires_grad(model, 1 + self.epoch // 2)
                if self.epoch == self.epoch_max:
                    self.keep_training = False
            if logging :
                logging.info("Finished Training")
            return results


    def training_loop(self):
        """

        :param model: model to train
        :param loader: training dataloader
        :param optimizer: optimizer
        :param criterion: criterion for the loss
        :param device: device to do the computations on
        :param minibatch_accumulate: number of minibatch to accumulate before applying gradient. Can be useful on smaller gpu memory
        :return: epoch loss, tensor of concatenated labels and predictions
        """
        running_loss = 0

        self.model.train()
        i = 1
        for images, labels in tqdm.tqdm(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            # send to GPU
            images, labels = (
                images.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )

            # Apply advanced transformation requiring multiple inputs

            images, labels = self.train_loader.dataset.advanced_transform((images, labels))
            images = self.train_loader.dataset.normalize(images)




            with torch.cuda.amp.autocast(enabled=self.autocast):
                if self.use_features :
                    features = self.model.module.forward_features(images).squeeze()
                outputs = self.model(images)
            if self.use_features:
                loss = self.criterion(features, labels)
            else :
                loss = self.criterion(outputs, labels)




            self.scaler.scale(loss).backward()
            #Unscales the gradients of optimizer's assigned params in-place
            if self.clip_norm !=0 :
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.epoch >= self.swa_start:
                self.model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.scheduler.step()

            running_loss += loss.detach()
            # ending loop

            #loader.iterable.dataset.step(idx.tolist(), outputs.detach().cpu().numpy())
            del (
                outputs,
                labels,
                images,
                loss,
            )  # garbage management sometimes fails with cuda
            i += 1

        if self.epoch >= self.swa_start and self.swa_start!=-1:
            torch.optim.swa_utils.update_bn(self.train_loader, self.model,device=self.device)
        return running_loss


    @torch.no_grad()
    def validation_loop(self):
        """

        :param model: model to evaluate
        :param loader: dataset loader
        :param criterion: criterion to evaluate the loss
        :param device: device to do the computation on
        :return: val_loss for the N epoch, tensor of concatenated labels and predictions
        """
        running_loss = 0

        self.model.eval()

        results = [torch.tensor([]), torch.tensor([])]

        for images, labels  in tqdm.tqdm(self.val_loader):
            # get the inputs; data is a list of [inputs, labels]

            # send to GPU
            images, labels = (
                images.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )


            images      = self.val_loader.dataset.normalize(images)

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=self.autocast):
                if self.use_features:
                    features = self.model.module.forward_features(images).squeeze()
                outputs = self.model(images)
            if self.use_features:
                loss = self.criterion(features, labels)
            else:
                loss = self.criterion(outputs, labels)

            running_loss += loss.detach()
            outputs = outputs.detach().cpu()
            results[1] = torch.cat((results[1], outputs), dim=0)
            results[0] = torch.cat((results[0], labels.cpu().round(decimals=0)),
                                   dim=0)  # round to 0 or 1 in case of label smoothing

            del (
                images,
                labels,
                outputs,
                loss,
            )  # garbage management sometimes fails with cuda

        return running_loss, results,









if __name__ == "__main__":
    experiment = Experiment(dir="/debug", names=["1", "2", "3"])
    os.environ["DEBUG"] = "True"

    experiment.log_metric("auc", {"banana": 0})

    experiment.log_metrics({"apple": 3})

    experiment.next_epoch(3, None)

    results = [torch.randint(0, 2, size=(10, 13)), torch.rand(size=(10, 13))]
    experiment.end(results)
