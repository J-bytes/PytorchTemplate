import numpy as np
import torch
import torch.distributed as dist
import tqdm
import torchattacks
from torchvision import transforms

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-07-19$

@author: Jonathan Beaulieu-Emond
"""


import tqdm

import torch


from PytorchTemplate.Experiment import Experiment

class AdversarialExperiment(Experiment):


    def training_loop(
            self
    ):
        """

        :param model: model to train
        :param loader: training dataloader
        :param optimizer: optimizer
        :param criterion: criterion for the loss
        :param device: device to do the computations on
        :param minibatch_accumulate: number of minibatch to accumulate before applying gradient. Can be useful on smaller gpu memory
        :return: epoch loss, tensor of concatenated labels and predictions
        """


        self.model.train()



        for i in range(2,16,2) :
            running_loss = 0
            eps = i / 255
            atk = torchattacks.APGD(self.model, eps=eps, steps=10)
            for images, labels in tqdm.tqdm(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)
                # send to GPU
                images, labels = (
                    images.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                )

                # Apply advanced transformation requiring multiple inputs

                images, labels = self.train_loader.dataset.advanced_transform((images, labels))

                images = atk(images, torch.argmax(labels,dim=1))

                images = images.to(self.device, non_blocking=True)
                with torch.no_grad():
                    images = self.train_loader.dataset.normalize(images)

                with torch.cuda.amp.autocast(enabled=self.autocast):

                    outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                if self.clip_norm != 0:
                    self.scaler.unscale_(self.optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.epoch > self.swa_start and self.swa_start != -1:
                    self.model.update_parameters(self.model)
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()

                running_loss += loss.detach()
                # ending loop

                # loader.iterable.dataset.step(idx.tolist(), outputs.detach().cpu().numpy())
                del (
                    outputs,
                    labels,
                    images,
                    loss,
                )  # garbage management sometimes fails with cuda

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
        eps=8/255
        atk = torchattacks.APGD(self.model, eps=eps, steps=100)
        for images, labels in tqdm.tqdm(self.val_loader):
            # get the inputs; data is a list of [inputs, labels]

            # send to GPU
            images, labels = (
                images.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )

            images = atk(images, torch.argmax(labels,dim=1))
            with torch.no_grad():
                    images = self.val_loader.dataset.normalize(images)

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=self.autocast):
                outputs = self.model(images)

            loss = self.criterion(outputs.float(), labels.float())

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



