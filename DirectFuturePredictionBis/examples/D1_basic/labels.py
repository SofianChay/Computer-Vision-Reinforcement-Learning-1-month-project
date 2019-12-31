WALL_ID = 0
FLOOR_CEILING_ID = 1
ITEM_ID = 2 

objects = [WALL_ID, FLOOR_CEILING_ID, ITEM_ID]

def transform_labels(img_labels):
	tmp = img_labels.copy()
	tmp[img_labels == ] = WALL_ID
	tmp[img_labels == ] = FLOOR_CEILING_ID
	tmp[img_labels == ] = ITEM_ID
	# ...
	return tmp