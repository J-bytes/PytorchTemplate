#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-02-20$

@author: Jonathan Beaulieu-Emond
"""

class Logger() :

    def __init__(self, wandb_run,verbose):
        self.wandb_run = wandb_run
        self.verbose = verbose

    def log(self, message):
        self.log_file.write(message)
        self.log_file.flush()

    def close(self):
        self.log_file.close()

if __name__ == "__main__":
    main()
