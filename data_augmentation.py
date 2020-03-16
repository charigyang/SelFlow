import tensorflow.compat.v1 as tf
import numpy as np

def random_crop(img_list, crop_h, crop_w):
    img_size = tf.shape(img_list[0])
    # crop image and flow
    rand_offset_h = tf.random_uniform([], 0, img_size[0]-crop_h+1, dtype=tf.int32)
    rand_offset_w = tf.random_uniform([], 0, img_size[1]-crop_w+1, dtype=tf.int32)
    
    for i, img in enumerate(img_list):
        img_list[i] = tf.image.crop_to_bounding_box(img, rand_offset_h, rand_offset_w, crop_h, crop_w)
    
    return img_list

def flow_vertical_flip(flow):
    flow = tf.image.flip_up_down(flow)
    flow_u, flow_v = tf.unstack(flow, axis=-1)
    flow_v = flow_v * -1
    flow = tf.stack([flow_u, flow_v], axis=-1)
    return flow

def flow_horizontal_flip(flow):
    flow = tf.image.flip_left_right(flow)
    flow_u, flow_v = tf.unstack(flow, axis=-1)
    flow_u = flow_u * -1
    flow = tf.stack([flow_u, flow_v], axis=-1)
    return flow

def random_flip(img_list):
    is_flip = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)
    
    for i in range(len(img_list)):
        img_list[i] = tf.where(is_flip[0] > 0, tf.image.flip_left_right(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_flip[1] > 0, tf.image.flip_up_down(img_list[i]), img_list[i])  
    return img_list

def random_flip_with_flow(img_list, flow_list):
    is_flip = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)
    for i in range(len(img_list)):
        img_list[i] = tf.where(is_flip[0] > 0, tf.image.flip_left_right(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_flip[1] > 0, tf.image.flip_up_down(img_list[i]), img_list[i]) 
    for i in range(len(flow_list)):
        flow_list[i] = tf.where(is_flip[0] > 0, flow_horizontal_flip(flow_list[i]), flow_list[i])
        flow_list[i] = tf.where(is_flip[1] > 0, flow_vertical_flip(flow_list[i]), flow_list[i])  
    return img_list, flow_list


def random_channel_swap(img_list):
    channel_permutation = tf.constant([[0, 1, 2],
                                       [0, 2, 1],
                                       [1, 0, 2],
                                       [1, 2, 0], 
                                       [2, 0, 1],
                                       [2, 1, 0]])    
    rand_i = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32)
    perm = channel_permutation[rand_i]
    for i, img in enumerate(img_list):
        channel_1 = img[:, :, perm[0]]
        channel_2 = img[:, :, perm[1]]
        channel_3 = img[:, :, perm[2]]
        img_list[i] = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    return img_list

def flow_resize(flow, out_size, is_scale=True, method=0):
    '''
        method: 0 mean bilinear, 1 means nearest
    '''
    flow_size = tf.to_float(tf.shape(flow)[-3:-1])
    flow = tf.image.resize_images(flow, out_size, method=method, align_corners=True)
    if is_scale:
        scale = tf.to_float(out_size) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        flow = tf.multiply(flow, scale)
    return flow

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image
def augment_L(image):
    rand_i = tf.random_uniform([], minval=0.95, maxval=1.05, dtype=tf.float32)
    return tf.stack([image[:,:,0]*rand_i, image[:,:,1], image[:,:,2]], axis=2)

def augment_a(image):
    rand_i = tf.random_uniform([], minval=0.95, maxval=1.05, dtype=tf.float32)
    return tf.stack([image[:,:,0]*1.5*rand_i, image[:,:,1]*0, image[:,:,2]*1.5], axis=2)

def augment_b(image):
    rand_i = tf.random_uniform([], minval=0.95, maxval=1.05, dtype=tf.float32)
    return tf.stack([image[:,:,0]*1.5*rand_i, image[:,:,1]*1.5, image[:,:,2]*0], axis=2)

def lab_augment(img_list):
    is_augment = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
    
    for i in range(len(img_list)):
        img_list[i] = tf.where(is_augment[0] == 1, augment_L(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_augment[0] == 2, augment_a(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_augment[0] == 3, augment_b(img_list[i]), img_list[i])  
    return img_list

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def convert_to_lab(img_list):
    img_size = tf.shape(img_list[0])

    for i, img in enumerate(img_list):
        img_list[i] = rgb_to_lab(img)
    
    return img_list