


# -*- coding: utf-8 -*-
import random
import torch
import math
import copy

class Downblock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding='same')
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.mp = torch.nn.MaxPool2d(kernel_size=2)
        self.activation = torch.nn.PReLU()

    def forward(self, x):
        self.input = copy.copy(x)  # backup the input for later
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp(x)
        x = self.activation(x)
        return x

class Upblock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same') # in the future replace with dilated conv?
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2) # to replace with transposed convolution
        self.activation = torch.nn.PReLU()

    def forward(self, x):
        input = copy.copy(x)  # backup the input for later
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.upsample(x)
        x = self.activation(x)
        return x,input


class Encoder(torch.nn.Module):
    def __init__(self,depth,channels):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()

        #downsampling to latent space
        self.downsampling=torch.nn.ModuleList([Downblock(in_channels=channels[i],out_channels=channels[i+1]) for i in range(depth-1)])
        self.inputs=[]
        self.depth=depth


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

        for d in range(0,self.depth) :
            self.inputs=self.inputs.append(x)
            x=self.downsampling[d](x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self,depth,channels,skips=[]):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.upsampling=torch.nn.ModuleList([Upblock(in_channels=channels[i],out_channels=channels[i+1]) for i in range(depth-1)])

        self.skips=skips[::-1]# inverse order
        self.depth=depth
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


        for d in range(0,self.depth):

            x = self.upsampling[d](x)
            x+=self.skips[d] #skip connection : to replace with concatenation later

        return x


class Unet(torch.nn.Module):
    def __init__(self, depth,channels):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        assert len(channels)==depth
        super().__init__()

        self.encoder = Encoder(depth, channels)
        self.decoder=Decoder(depth,channels[::-1])
        self.depth=depth

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

        x=self.encoder(x)
        self.decoder.skips=self.encoder.inputs[::-1]
        x=self.decoder(x)
        return x






