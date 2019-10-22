from test_datasets import BasicDataset
import os
from PIL import Image
from skimage.segmentation import slic
import numpy as np
import cv2
data_list_file = "./dataset/KITTI/train_raw_2015_with_id.txt"
img_dir = "../datasets/KITTI/training"
save_dir = "../datasets/KITTI_superpix/training"
dataset = BasicDataset(data_list_file=data_list_file, img_dir=img_dir, is_normalize_img=False)
load_list = dataset.data_list[:, 2]
save_list = load_list #[s.replace('.png', '.npy') for s in load_list]
for l,s in zip(load_list,save_list):
	print(l)
	im = Image.open(os.path.join(img_dir,l))
	seg = slic(im)
	#np.save(os.path.join(save_dir,s),seg)
	cv2.imwrite(os.path.join(save_dir,s), seg)
	print(seg)
	print(s)

        """        
        superpixels = np.load(self.superpixel_dir + "/image_2/000000_02.npy")
        superpixels = tf.convert_to_tensor(superpixels, dtype=tf.int32)
        """