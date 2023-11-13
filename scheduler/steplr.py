#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
	
	# sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
	# 													mode=mode, 
	# 													factor=factor, 
	# 													patience=patience, 
	# 													threshold=threshold, 
	# 													threshold_mode=threshold_mode, 
	# 													cooldown=cooldown, 
	# 													min_lr=min_lr, 
	# 													eps=eps)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step


# import torch

# def Scheduler(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, **kwargs):
#     sche_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
#     print('Initialised ReduceLROnPlateau scheduler')
#     return sche_fn

# def Optimizer(parameters, lr, weight_decay, **kwargs):
#     print('Initialised Adam optimizer')
#     return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

# # Example usage
# optimizer = Optimizer(model.parameters(), lr=0.001, weight_decay=1e-5)
# scheduler = Scheduler(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
