import os
import random

home_dir = '/ssd/charig/UCF-101_png'
labels = os.listdir(home_dir)
for label in labels:
	videos = os.listdir(os.path.join(home_dir,label))
	videos = random.sample(videos, 10)
	for video in videos:
		images = os.listdir(os.path.join(home_dir,label,video))
		num_images = len(images)
		image_dir = os.path.join(label,video)
		#i_s = random.sample(range(1, num_images-9), )
		for i in range(num_images-9):
			print(" ".join([os.path.join(image_dir,images[i]),
							os.path.join(image_dir,images[i+2]),
							os.path.join(image_dir,images[i+4]),
							os.path.join(image_dir,images[i+6]),
							os.path.join(image_dir,images[i+8]),
							str(i+1).zfill(5),
							#os.path.join(image_dir,images[i+3]),
							#os.path.join(image_dir,images[i+4]),
							]))
		#raise Exception
# we want label/video/frame *5
