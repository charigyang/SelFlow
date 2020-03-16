import os
import random

home_dir = '/ssd/charig/UCF-101_jpg'
labels = os.listdir(home_dir)
for label in labels[0:10]:
	videos = os.listdir(os.path.join(home_dir,label))
	video = videos[0]
	images = os.listdir(os.path.join(home_dir,label,video))
	num_images = len(images)
	image_dir = os.path.join(label,video)
	for i in range(num_images-3):
		print(" ".join([os.path.join(image_dir,images[i]),
						os.path.join(image_dir,images[i+1]),
						os.path.join(image_dir,images[i+2]),
						]))
# we want label/video/frame *5