from vizdoom import *
import random
import time
import numpy as np 
import skimage.color, skimage.transform
import argparse
import matplotlib.pyplot as plt
import random

# ARGUMENTS 
parser = argparse.ArgumentParser()
parser.add_argument("--scenario", default='basic.cfg', help="Choose the vizdoom scenario to generate. This is a (.cfg file). default is basic.cfg")
args = parser.parse_args()


# PREPROCESSING
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

resolution = (200, 200, 3)


# GROUND TRUTHS SEGMENTATION
def color_labels(labels):
    tmp = np.stack([labels] * 3, -1)
    tmp[labels == 0] = [255, 0, 0]
    tmp[labels == 1] = [0, 0, 255]
    return tmp


# INIT GAME 
# ticrate = 1000
game = DoomGame()
game.load_config("scenarios/"+args.scenario)

game.set_window_visible(False)
game.set_mode(Mode.PLAYER)
# Enables labeling of the in game objects.
game.set_labels_buffer_enabled(True)
# game.set_ticrate(ticrate)
game.set_screen_resolution(ScreenResolution.RES_320X240) # RES_320X240, RES_640X480, RES_1920X1080
game.set_screen_format(ScreenFormat.CRCGCB) # grayscale image  CRCGCB, RGB24, GRAY8, BGR24

game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]


episodes = 5

if __name__ == '__main__':

    input_images = list()
    output_images = list()

    for i in range(episodes):
        game.new_episode()
        print(f'episode {i + 1} ...')
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            labels = state.labels_buffer
            # retrieve data :
            input_images.append(preprocess(np.transpose(img, axes=(1, 2, 0)))) # dim (3, 240, 320) -> (240, 320, 3)
            output_images.append(preprocess(color_labels(labels)))
            game.make_action(random.choice(actions))

       
    game.close()

    # SHOW SOME EXAMPLES 

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 4))
    for k in range(5):
        tmp = random.choice(range(len(input_images)))
        axes[k][0].imshow(input_images[tmp])
        axes[k][1].imshow(output_images[tmp])

    fig.tight_layout()
    plt.show()