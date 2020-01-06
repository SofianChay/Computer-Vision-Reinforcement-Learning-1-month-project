from .U_Net import UNet
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch import nn
import numpy as np
import pickle
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
# training functions
def training_step(model, batch_X, batch_y, criterion, optimizer, width_out, height_out, n_classes):
	inputs = batch_X.to(device)
	truths = batch_y.to(device)
	optimizer.zero_grad()
	with torch.set_grad_enabled(True):
		outputs = model(inputs.float())
		outputs = outputs.permute(0, 2, 3, 1)
		# outputs.shape = (batch_size, n_classes, img_cols, img_rows)
		m = outputs.shape[0]      
		outputs = outputs.view(m*width_out*height_out, n_classes)
		# outputs.shape =(batch_size, img_cols, img_rows, n_classes)
		truths = truths.view(m*width_out*height_out)
		loss = criterion(outputs, truths.long())
		loss.backward()
		optimizer.step()
	return loss

def validation_step(model, criterion, val_X, val_y, width_out, height_out, n_classes):
	inputs = val_X.to(device)
	truths = val_y.to(device)
	with torch.no_grad():
		outputs = model(inputs.float())
		outputs = outputs.permute(0, 2, 3, 1)
		# outputs.shape = (batch_size, n_classes, img_cols, img_rows)
		m = outputs.shape[0]
		outputs = outputs.view(m*width_out*height_out, n_classes)
		# outputs.shape = (batch_size, img_cols, img_rows, n_classes)
		truths = truths.view(m*width_out*height_out)
		loss = criterion(outputs, truths.long())
	return loss

def train(model, criterion, optimizer, num_epochs, data, algo, width_out, height_out, n_classes, test):
	len_train = len(data['train']['X'])
	len_val = len(data['val']['X'])
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		train_loss = 0
		val_loss = 0
		for phase in ['train', 'val']:
			if phase == 'train':
				print("learning")
				model.train()
				for batch_X, batch_y in tqdm(zip(data['train']['X'], data['train']['y'])):
					train_loss += training_step(model, batch_X, batch_y, criterion, optimizer, width_out, height_out, n_classes)
				print(f'train loss = {round(float(train_loss / len_train), 4)} ')
			else:
				if test:
					print("validation")
					model.eval()
					for val_X, val_y in zip(data['val']['X'], data['val']['y']):
						val_loss += validation_step(model, criterion, val_X, val_y, width_out, height_out, n_classes)
					print(f'validation loss = {round(float(val_loss / len_val), 4)}') 
	# torch.save(model, algo)
	# print('model saved to ' + algo)
##########################

