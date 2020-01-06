# taken from vision for action

import numpy as np


WALL_ID = 0
FLOOR_CEILING_ID = 1

FLOOR = 0
ENEMY = 1
OTHER = 2
WALL = 3
ITEM = 4
CEILING = 5
N_SEMANTICS = 6 

DEBUG_COLORS = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        ]

NAME_TO_LABEL = dict()
NAME_TO_LABEL['DoomPlayer'] = OTHER
NAME_TO_LABEL['ClipBox'] = ITEM
NAME_TO_LABEL['RocketBox'] = ITEM
NAME_TO_LABEL['CellPack'] = ITEM
NAME_TO_LABEL['RocketLauncher'] = ITEM
NAME_TO_LABEL['Stimpack'] = ITEM
NAME_TO_LABEL['Medikit'] = ITEM
NAME_TO_LABEL['HealthBonus'] = ITEM
NAME_TO_LABEL['ArmorBonus'] = ITEM
NAME_TO_LABEL['GreenArmor'] = ITEM
NAME_TO_LABEL['BlueArmor'] = ITEM
NAME_TO_LABEL['Chainsaw'] = ITEM
NAME_TO_LABEL['PlasmaRifle'] = ITEM
NAME_TO_LABEL['Chaingun'] = ITEM
NAME_TO_LABEL['ShellBox'] = ITEM
NAME_TO_LABEL['SuperShotgun'] = ITEM
NAME_TO_LABEL['TeleportFog'] = OTHER
NAME_TO_LABEL['Zombieman'] = ENEMY
NAME_TO_LABEL['ShotgunGuy'] = ENEMY
NAME_TO_LABEL['HellKnight'] = ENEMY
NAME_TO_LABEL['MarineChainsawVzd'] = ENEMY
NAME_TO_LABEL['BaronBall'] = ENEMY
NAME_TO_LABEL['Demon'] = ENEMY
NAME_TO_LABEL['ChaingunGuy'] = ENEMY
NAME_TO_LABEL['Blood'] = OTHER
NAME_TO_LABEL['Clip'] = ITEM
NAME_TO_LABEL['Shotgun'] = ITEM

NAME_TO_LABEL['CustomMedikit'] = ITEM
NAME_TO_LABEL['DoomImp'] = ENEMY
NAME_TO_LABEL['DoomImpBall'] = ENEMY
NAME_TO_LABEL['BulletPuff'] = OTHER
NAME_TO_LABEL['Poison'] = ENEMY

NAME_TO_LABEL['BurningBarrel'] = OTHER
NAME_TO_LABEL['ExplosiveBarrel'] = OTHER
NAME_TO_LABEL['DeadExplosiveBarrel'] = OTHER
NAME_TO_LABEL['Column'] = OTHER
NAME_TO_LABEL['ShortGreenTorch'] = OTHER


def transform_labels(labels, labels_buffer):
    semantics = np.zeros(labels_buffer.shape, dtype=np.uint8)
    n = labels_buffer.shape[1] // 2
    semantics[labels_buffer==WALL_ID] = WALL
    semantics[labels_buffer==FLOOR_CEILING_ID] = FLOOR
    tmp = np.zeros(labels_buffer.shape, dtype=np.uint8)
    tmp[:, n:] = 1
    semantics[(labels_buffer==FLOOR)*tmp] = CEILING
    for label in labels:
        obj = NAME_TO_LABEL.get(label.object_name, OTHER)
        semantics[labels_buffer==label.value] = obj
    return semantics


def labels_to_rgb(label_frame):
    rgb_array = np.zeros(label_frame.shape + (3,), dtype=np.uint8)
    for i in range(N_SEMANTICS):
        rgb_array[label_frame==i, :] = DEBUG_COLORS[i]
    return rgb_array
