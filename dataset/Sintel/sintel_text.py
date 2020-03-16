import os
import random

home_dir = '/data/charig/datasets/Sintel/training'
labels = os.listdir(home_dir)
j=0
for label in labels:
	videos = [os.path.join(home_dir,'clean'), os.path.join(home_dir,'final')]
	print(videos)
	for video in videos:
		images = os.listdir(os.path.join(home_dir,label,video))
		num_images = len(images)
		image_dir = os.path.join(label,video)
		#i_s = random.sample(range(1, num_images-5), 10)
		for i in range(num_images-5):
			print(" ".join([os.path.join(image_dir,images[i]),
							os.path.join(image_dir,images[i+1]),
							os.path.join(image_dir,images[i+2]),
							os.path.join(image_dir,images[i+3]),
							os.path.join(image_dir,images[i+4]),
							str(j+1).zfill(5),
							]))
			j+=1
# we want label/video/frame *5
