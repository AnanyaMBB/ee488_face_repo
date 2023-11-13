#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_size=test_interval, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised Expeonential LR scheduler')

	return sche_fn, lr_step