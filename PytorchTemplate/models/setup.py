#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-21$

@author: Jonathan Beaulieu-Emond
"""
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='relative_attention_cpp',
      ext_modules=[cpp_extension.CppExtension('relative_attention_cpp', ['relative_attention.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})