from vizdoom import *
import random
import time
import numpy as np 
import skimage.color, skimage.transform
import argparse
import matplotlib.pyplot as plt
import random
import torch
import pickle


# ARGUMENTS 
parser = argparse.ArgumentParser()
parser.add_argument("--scenario", default='basic.cfg', help="Choose the vizdoom scenario to generate. This is a (.cfg file). default is basic.cfg")
parser.add_argument("--vizualize", default=False, help="vizualize examples", type=bool)
args = parser.parse_args()
batch_size = 1

# PREPROCESSING
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    # img = img.astype(np.float32)
    return img

resolution = (1, 146, 226)


# GROUND TRUTHS SEGMENTATION
def color_labels(labels):
    # tmp = np.stack([labels] * 3, -1)
    # tmp[labels == 0] = [255, 0, 0]
    # tmp[labels == 1] = [0, 0, 255]
    # return tmp
    labels[labels == 0] = 0
    labels[labels == 170] = 1
    labels[labels == 127] = 2
    labels[labels == 255] = 3
    labels[labels == 85] = 4
    labels[labels == 63] = 5
    labels[labels == 191] = 6
    return labels


# INIT GAME 
# ticrate = 1000    
game = DoomGame()
game.load_config("scenarios/"+args.scenario)

game.set_window_visible(False)
game.set_render_hud(False)
game.set_mode(Mode.PLAYER)
# Enables labeling of the in game objects.
game.set_labels_buffer_enabled(True)
# game.set_ticrate(ticrate)
game.set_screen_resolution(ScreenResolution.RES_320X240) # RES_320X240, RES_640X480, RES_1920X1080
game.set_screen_format(ScreenFormat.GRAY8) # grayscale image  CRCGCB, RGB24, GRAY8, BGR24

game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]


episodes_train = 5
episodes_val = 1

if __name__ == '__main__':

    # create dict of tensors. validation and train / use pickle file (for the moment)
    data = {'train': {'X': list(), 'y': list()}, 'val': {'X': list(), 'y': list()}}
    train_X = list()
    train_y = list()

    # create train set 
    print('creating train set ...')
    for i in range(episodes_train):
        game.new_episode()
        print(f'episode {i + 1} running ...')
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            labels = state.labels_buffer
            # retrieve data :
            train_X.append(torch.from_numpy(img.reshape((1, img.shape[0], img.shape[1])).astype(np.float))) 
            train_y.append(torch.from_numpy(color_labels(labels).reshape((1, img.shape[0], img.shape[1]))))
            game.make_action(random.choice(actions))


        # create val set 
    print('\ncreating val set ...')
    for i in range(episodes_val):
        game.new_episode()
        print(f'episode {i + 1} running ...')
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            labels = state.labels_buffer
            # retrieve data :
            data['val']['X'].append(torch.from_numpy(img.reshape((1, 1, img.shape[0], img.shape[1])).astype(np.float)))
            data['val']['y'].append(torch.from_numpy(color_labels(labels).reshape((1, 1, img.shape[0], img.shape[1]))))
            game.make_action(random.choice(actions))
       
    game.close()

    # create batches
    print('\ncreating batches ...')
    indices = list(range(len(train_X)))
    random.shuffle(indices)
    for i in range(len(train_X)//batch_size):
        batch_X = list()
        batch_y = list()
        for j in range(batch_size):
            batch_X.append(train_X[indices[i*batch_size + j]])
            batch_y.append(train_y[indices[i*batch_size + j]])
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y)
        data['train']['X'].append(batch_X)
        data['train']['y'].append(batch_y)




#    for put in ['X', 'y']:
#        data['val'][put] = torch.stack(data['val'][put])


    # save into pickle file
    pickle.dump(data, open('data', 'wb'))


    # SHOW SOME EXAMPLES
    if args.vizualize :
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))
        for k in range(5):
            tmp = random.choice(range(len(data['val']['X'])))
            axes[k][0].imshow(data['val']['X'][tmp][0, :])
            axes[k][1].imshow(data['val']['y'][tmp][0, :])

        fig.tight_layout()
        plt.show()






