#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-22$

@author: Jonathan Beaulieu-Emond
"""
import torch
@torch.jit.script
class CutOut :
    def __init__(self,p : float=0.2,intensity : float=0.4) :
        self.p = p
        self.intensity = (1-intensity)/2
        assert 0 <= self.intensity <= 0.5
        assert 0 <= self.p <= 1


    def __call__(self, inputs):
        if self.p == 0:
            return (inputs[0],inputs[1])

        images, labels = inputs[0],inputs[1]
        N, C, H, W = images.shape


        idx = torch.rand(size=(N,)) < self.p
        M = int(torch.sum(idx).item())
        idx2 = torch.randint(low=0, high=len(images), size=[M,]).long()
        x11 = torch.randint(low=0, high=int((1-self.intensity) * H), size=[M,]).long()
        x22 = torch.randint(low=int(self.intensity * H),high= H, size=[M,]).long()

        x1 = torch.minimum(x11,x22)
        x2 = torch.maximum(x11,x22)

        y11 = torch.randint(0, int((1-self.intensity) * W), size=[M,]).long()
        y22 = torch.randint(int(self.intensity * W), W, size=[M,]).long()

        y1 = torch.minimum(y11,y22)
        y2 = torch.maximum(y11,y22)



        for idxx,(x11, x22, y11, y22, apply, idx22) in enumerate(zip(x1, x2, y1, y2, idx, idx2)):
            if apply:
                images[idxx, :, x11:x22, y11:y22] = torch.clamp(torch.normal(mean=0.5,std=1.,size=(1,C,int(x22-x11),int(y22-y11))),0,1)

        return (images,labels)


if __name__=="__main__" :

    images,labels = CutOut()((torch.rand(32,3,32,32),torch.rand(2,10)))

