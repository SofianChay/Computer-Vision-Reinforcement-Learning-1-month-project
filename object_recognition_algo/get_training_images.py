from vizdoom import *
import random
import time
import numpy as np 
import skimage.color, skimage.transform
import matplotlib.pyplot as plt
import random
import torch


# GROUND TRUTHS SEGMENTATION
def encode(labels, labels_figures):
    for i in range(len(labels_figures)):
        labels[labels == labels_figures[i]] = i
    return labels


def define_actions(game):
    actions = list()
    buttons = len(game.get_available_buttons())
    for i in range(buttons):
        button = [1 if j == i else 0 for j in range(buttons)]
        actions.append(button)
    return actions


def generate_examples(scenario, visualize, batch_size, episodes_train, episodes_val):
    
    # INIT GAME  
    game = DoomGame()
    game.load_config("scenarios/" + scenario)

    game.set_window_visible(False)
    game.set_render_hud(False)
    game.set_mode(Mode.PLAYER)
    # Enables labeling of the in game objects.
    game.set_labels_buffer_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_320X240) # RES_320X240, RES_640X480, RES_1920X1080
    game.set_screen_format(ScreenFormat.GRAY8) # grayscale image  CRCGCB, RGB24, GRAY8, BGR24

    game.init()

    actions = define_actions(game)
    
    
    # create dict of tensors. validation and train / use pickle file (for the moment)
    data = {'train': {'X': list(), 'y': list()}, 'val': {'X': list(), 'y': list()}}
    train_X = list()
    train_y = list()

    # create train set 
    print('creating train set ...')
    labels_figures = [0]
    for i in range(episodes_train):
        game.new_episode()
        print(f'episode {i + 1} running ...')
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            labels = state.labels_buffer
            for label in state.labels:
                if label.value not in labels_figures:
                    labels_figures.append(label.value)
            # retrieve data :
            train_X.append(torch.from_numpy(img.reshape((1, img.shape[0], img.shape[1])).astype(np.float))) 
            train_y.append(torch.from_numpy(encode(labels, labels_figures).reshape((1, img.shape[0], img.shape[1]))))
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
            data['val']['y'].append(torch.from_numpy(encode(labels, labels_figures).reshape((1, 1, img.shape[0], img.shape[1]))))
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

    # SHOW SOME EXAMPLES
    if visualize :
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))
        for k in range(5):
            tmp = random.choice(range(len(data['val']['X'])))
            axes[k][0].imshow(data['val']['X'][tmp][0, 0, :])
            axes[k][1].imshow(data['val']['y'][tmp][0, 0, :])
        fig.tight_layout()
        plt.show()

    
    return data, labels_figures






