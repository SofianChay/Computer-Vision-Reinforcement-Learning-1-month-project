from U_Net import UNet
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch import nn
import numpy as np
import pickle


# gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################
# training functions
def training_step(model, batch_X, batch_y, criterion, optimizer, width_out, height_out, n_classes):
	inputs = batch_X.to(device)
	truths = batch_y.to(device)
	optimizer.zero_grad()
	with torch.set_grad_enabled(True):
		outputs = model(inputs)
		outputs = outputs.permute(0, 2, 3, 1)
		# outputs.shape = (batch_size, n_classes, img_cols, img_rows)
		m = outputs.shape[0]
		outputs = outputs.resize(m*width_out*height_out, n_classes)
		# outputs.shape =(batch_size, img_cols, img_rows, n_classes)
		truth = truth.resize(m*width_out*height_out)
		loss = criterion(outputs, truths)
		loss.backward()
		optimizer.step()
	return loss

def validation_step(model, criterion, data_test, width_out, height_out, n_classes):
	inputs = data_test['X'].to(device)
	truths = data_test['y'].to(device)
	with torch.set_grad_enabled(False):
		outputs = model(inputs)
		outputs = outputs.permute(0, 2, 3, 1)
		# outputs.shape = (batch_size, n_classes, img_cols, img_rows)
		m = outputs.shape[0]
		outputs = outputs.resize(m*width_out*height_out, n_classes)
		# outputs.shape = (batch_size, img_cols, img_rows, n_classes)
		truth = truth.resize(m*width_out*height_out)
		loss = criterion(outputs, truths)
	return loss/len(data_test)

def train(model, criterion, optimizer, num_epochs, data, algo, width_out, height_out, n_classes):
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		train_loss = 0
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
				for batch_X, batch_y in zip(data['train']['X'], data['train']['y']):
					train_loss += training_step(model, batch_X, batch_y, criterion, optimizer, width_out, height_out, n_classes)
				print(f'train loss = {train_loss/len(data[train])}')
			else:
				model.eval()
				val_loss = validation_step(model, criterion, data['val'], width_out, height_out, n_classes)
				print(f'validation loss = {val_loss}')
	torch.save(model, algo + '.pth')
	print('model saved to ' + algo + '.pth')
##########################


##########################
# def plot_examples(model, datax, datay, num_examples):
# 	fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 4*num_examples))
# 	m = datax.shape[0]
# 	for row_num in range(num_examples):
# 		image_indx = np.random.randint(m)
# 		image_arr = model(torch.from_numpy(datax[image_indx:image_indx + 1]).float().cuda()).squeeze(0).detach.cpu().numpy()
# 		ax[row_num][0].imshow(np.transpose(datax[image_indx], (1, 2, 0))[:, :, 0]) # show input 1st channel
# 		ax[row_num][1].imshow(np.transpose(image_arr, (1, 2, 0))[:, :, 0]) # show result of unet
# 		ax[row_num][2].imshow(image_arr.argmax(0)) # show argmax of result of unet
# 		ax[row_num][3].imshow(np.transpose(datay[image_indx], (1, 2, 0))[:, :, 0]) # show ground truth
# 	plt.show()
#########################


#########################
def main():
	# data
	data = pickle.load(open('../ground_truth_generator/data', 'rb'))
	width_in = data['val']['X'][0].shape[1]
	height_in = data['val']['X'][0].shape[2]
	width_out = width_in
	height_out = height_in
	n_classes = len(np.unique(data['val']['y'][0]))
	# parameters
	num_epochs = 20
	# model
	in_channel =  1 # 1 if gray scale, 3 if RGB
	out_channel = n_classes # number of segments  (depends on the environment) should be len(np.unique(labels))
	model = UNet(in_channel, out_channel).to(device)
	# criterion
	criterion = nn.CrossEntropyLoss()
	# optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.99)
	# algo
	algo = 'segmentation'
	# train 
	train(model, criterion, optimizer, num_epochs, data, algo, width_out, height_out, n_classes)


if __name__ == '__main__':
	main()