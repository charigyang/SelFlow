
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

rgb_location = "/ssd/charig/DAVIS/JPEGImages/Full-Resolution"
pwc_location = "/ssd/charig/DAVIS_flo"
self_sup_location = "/data/charig/SelFlow/Davis_val/from_pretrained"
self_unsup_location = "/data/charig/SelFlow/Davis_val/occ"
data_list_file = "/data/charig/SelFlow/dataset/Davis/davis_val.txt"

dirs = os.listdir("comparison")
for name in dirs:
	print("/ssd/charig/ffmpeg-git-20200119-amd64-static/ffmpeg -r 5 -i comparison/{}/frames/{}_%03d.png -qscale 0 -vcodec mpeg4 -y comparison_cam/{}.mp4".format(name,name,name))
	#print("rm -r comparison/{}".format(name))
	print(" ")
	
	#os.system("/ssd/charig/ffmpeg-git-20200119-amd64-static/ffmpeg -r 20 -i \%05d.jpg -qscale 0 -vcodec mpeg4 -y ../comparison_video/{}.mp4".format(name))

"""
python video2.py > test.txt
chmod +x test.txt
./test.txt
"""