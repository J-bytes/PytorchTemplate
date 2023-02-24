#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-22$

@author: Jonathan Beaulieu-Emond
"""
import torch

@torch.jit.script
class MixUp:
    def __init__(self, p: float = 0.2, intensity: float = 0.4):
        self.p = p
        self.intensity = intensity
        assert 0 <= self.intensity <= 1
        assert 0 <= self.p <= 1

    def __call__(self, inputs):
        if self.p == 0:
            return (inputs[0], inputs[1])

        images, labels = inputs[0],inputs[1]
        N, C, H, W = images.shape
        # mixing
        idx = torch.rand(size=(N,)) < self.p
        M = int(torch.sum(idx).item())
        idx2 = torch.randint(low=0, high=len(images), size=[M, ]).long()
        images[idx] = (1-self.intensity) * images[idx] + self.intensity * images[idx2]
        labels[idx] = (1-self.intensity) * labels[idx] + self.intensity * labels[idx2]

        return (images,labels)