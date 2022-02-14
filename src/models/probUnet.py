# -*- coding: utf-8 -*-
import random
import torch
import math


class Encoder(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()

        #downsampling to latent space
        self.conv1=
        self.bn1

        self.conv2
        self.bn2

        self.conv3
        self.bn3


        #upsampling
        self.conv4 =
        self.bn4

        self.conv5
        self.bn5

        self.conv6
        self.bn6
        self.activation=torch.nn.PReLU()



    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """


        x=self.conv1(x)
        x=self.activation(x)
        x=self.bn1(x)

        x=self.conv2(x)
        x = self.activation(x)
        x=self.bn2(x)

        x=self.conv3(x)
        x = self.activation(x)
        x = self.bn3(x)

        return x



