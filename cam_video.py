
"""
TODO: 

load RGB
load PWC-Flow
load SelFlow-sup
load Selflow-unsup

resize them

stitch each image

combine into video
/ssd/charig/ffmpeg-git-20200119-amd64-static/ffmpeg -r 20 -i %05d.jpg -qscale 0 -vcodec mpeg4 -y movie.mp4
"""
from PIL import Image
import os
import cv2

rgb_location = "/ssd/charig/CamouflagedAnimalDataset"
pwc_location = "/ssd/charig/Cam_flo"
self_sup_location = "/data/charig/SelFlow/Camouflage_images/sup/color" #train/from sintel
self_unsup_location = "/data/charig/SelFlow/Camouflage_images/occ/color"
data_list_file = "/data/charig/SelFlow/dataset/Camouflage/cam.txt"

def merge_images(im1, im2, im3, im4):
	#rgb, pwc, self_sintel, self_self
	image1 = Image.open(im1)
	image2 = Image.open(im2)
	image3 = Image.open(im3)
	image4 = Image.open(im4)
	
	(w, h) = image1.size
	w_ = int(w)
	h_ = int(h)
	newsize = (w_, h_)
	image1 = image1.resize(newsize)
	image2 = image2.resize(newsize)
	image3 = image3.resize(newsize)
	image4 = image4.resize(newsize)

	result = Image.new('RGB', (w_*2, h_*2))
	result.paste(im=image1, box=(0, 0))
	result.paste(im=image2, box=(w_, 0))
	result.paste(im=image3, box=(0, h_))
	result.paste(im=image4, box=(w_, h_))

	return result

with open(data_list_file, 'rb') as f:
	for line in f:
		name = repr(line.split()[2])[2:-1]
		idx = repr(line.split()[-1])[2:-1]
		print(name)
		print(idx)
		rgb = os.path.join(rgb_location,name)
		pwc = os.path.join(pwc_location,name)
		self_sup = os.path.join(self_sup_location, name)
		self_unsup = os.path.join(self_unsup_location, name)
		merged_image = merge_images(rgb,pwc,self_sup,self_unsup)
		if not os.path.exists("comparison/" + name[:-10]):
			os.makedirs("comparison/" + name[:-10])
		merged_image.save("comparison/" + name)
