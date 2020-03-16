import os

home_dir = '/ssd/charig/DAVIS/JPEGImages/Full-Resolution'
train_list = '/ssd/charig/DAVIS/ImageSets/2017/val.txt' #train
labels = os.listdir(home_dir)
labels.sort()
j=0

with open(train_list) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
labels = [x.strip() for x in content] 

for label in labels:
	images = os.listdir(os.path.join(home_dir,label))
	images.sort()
	num_images = len(images)
	image_dir = label
	for i in range(num_images-5):
		j+=1
		print(" ".join([os.path.join(image_dir,images[i]),
						os.path.join(image_dir,images[i+1]),
						os.path.join(image_dir,images[i+2]),
						os.path.join(image_dir,images[i+3]),
						os.path.join(image_dir,images[i+4]),
						str(j).zfill(5),
						]))