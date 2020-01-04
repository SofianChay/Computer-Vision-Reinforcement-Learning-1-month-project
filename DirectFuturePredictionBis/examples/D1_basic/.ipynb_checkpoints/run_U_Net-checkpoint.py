from U_Net import UNet
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch import nn
import numpy as np
import pickle
from get_training_images import generate_examples
import argparse

##########################
# argments
# parser = argparse.ArgumentParser()
# parser.add_argument("--train_model", default=True, help="Type False if you want to use a pretrained model (a .pth file is necessary)", type=bool)
# parser.add_argument("--algo", default="segmentation", help="segmentation, depth_detection or optical_flow")
# parser.add_argument("--scenario", default='basic.cfg', help="Choose the vizdoom scenario to generate. This is a (.cfg file). default is basic.cfg")
# parser.add_argument("--visualize", default=False, help="Type True if you want to visualize examples", type=bool)
# parser.add_argument("--batch_size", default=1, help="Choose batch size", type=int)
# parser.add_argument("--episodes_train", default=10, help="Choose number of episodes to learn the model", type=int)
# parser.add_argument("--episodes_val", default=3, help="Choose number of episodes to test the model", type=int)
# args = parser.parse_args()

##########################
# gpu
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
	torch.save(model, algo)
	print('model saved to ' + algo)
##########################


##########################
def plot_examples(model, datax, datay, num_examples, labels_figures):
	model.eval()
	fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4*num_examples))
	m = len(datax)
	for row_num in range(num_examples):
		image_indx = np.random.randint(m)
		image_arr = model(datax[image_indx].to(device).float())
		image_arr = image_arr.squeeze(0).detach().cpu().numpy()
		ax[row_num][0].imshow(datax[image_indx][0, 0, :, :])  # show input 1st channel
		ax[row_num][1].imshow(decode(image_arr.argmax(0), labels_figures))  # show argmax of result of unet
		ax[row_num][2].imshow(decode(datay[image_indx][0, 0, :, :], labels_figures))  # show ground truth
	plt.show()
    
def decode(output, labels_figures):
    for i in range(len(labels_figures)):
        output[output == i] = labels_figures[i]
    return output
#########################


#########################
def main(scenario, visualize, batch_size, episodes_train, episodes_val, algo, train_model):
	# data
	print("loading data")
	data, labels_figures = generate_examples(scenario, visualize, batch_size, episodes_train, episodes_val)
	width_in = data['train']['X'][0].shape[2]
	height_in = data['train']['X'][0].shape[3]
	print(f'input size : {height_in}, {width_in}')
	width_out = data['train']['y'][0].shape[2]
	height_out = data['train']['y'][0].shape[3]
	print(f'output size : {height_out}, {width_out}')
	n_classes = len(labels_figures)
	print(f'number of objects to identify : {n_classes}')
	# parameters
	num_epochs = 1
	# model
	in_channel =  1 # 1 if gray scale, 3 if RGB
	out_channel = n_classes # number of segments  (depends on the environment) should be len(np.unique(labels)
	# train
	if train_model:    
		model = UNet(in_channel, out_channel).to(device)
		optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.99)
		criterion = nn.CrossEntropyLoss()
		train(model, criterion, optimizer, num_epochs, data, algo, width_out, height_out, n_classes, episodes_val)
	else:
		print('loading pre-trained model...')
		model = torch.load(algo)        
	if episodes_val:
		print("showing examples :")
		plot_examples(model, data['val']['X'], data['val']['y'], 5, labels_figures)
	return model 


# if __name__ == '__main__':
# 	main(args.scenario, args.visualize, args.batch_size, args.episodes_train, args.episodes_val, args.algo, args.train_model)
