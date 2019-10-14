# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow.compat.v1 as tf
import numpy as np
import os
import sys
import time
import cv2

from six.moves import xrange
from scipy import misc, io
from tensorflow.contrib import slim

import matplotlib.pyplot as plt
from network import pyramid_processing, pyramid_processing_five_frame, get_shape
from datasets import BasicDataset
from utils import average_gradients, lrelu, occlusion, rgb_bgr
from data_augmentation import flow_resize
from flowlib import flow_to_color, write_flo
from warp import tf_warp

class SelFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_self_supervision", 
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, self_supervision_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1       
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.self_supervision_config = self_supervision_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)         
        
        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))         
            
        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)  
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))    
        
        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir) 
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train']))) 
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))             

    def create_dataset_and_iterator(self, training_mode='no_self_supervision'):
        if training_mode=='no_self_supervision':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'])
            iterator = dataset.create_batch_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)   
        elif training_mode == 'self_supervision':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'],
                   fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
            iterator = dataset.create_batch_distillation_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads) 
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_self_supervision, self_supervision}')
        return dataset, iterator

    def epe_loss(self, diff, mask):
        diff_norm = tf.norm(diff, axis=-1, keepdims=True)
        diff_norm = tf.multiply(diff_norm, mask)
        diff_norm_sum = tf.reduce_sum(diff_norm)
        loss_mean = diff_norm_sum / (tf.reduce_sum(mask) + 1e-6)
        
        return loss_mean 
    
    def abs_robust_loss(self, diff, mask, q=0.4):
        diff = tf.pow((tf.abs(diff)+0.01), q)
        diff = tf.multiply(diff, mask)
        diff_sum = tf.reduce_sum(diff)
        loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6) 
        return loss_mean 
    
    def create_mask(self, tensor, paddings):
        with tf.variable_scope('create_mask'):
            shape = tf.shape(tensor)
            inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
            inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
            inner = tf.ones([inner_width, inner_height])
    
            mask2d = tf.pad(inner, paddings)
            mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
            mask4d = tf.expand_dims(mask3d, 3)
            return tf.stop_gradient(mask4d) 

    def census_loss(self, img1, img2_warped, mask, max_distance=3):
        patch_size = 2 * max_distance + 1
        with tf.variable_scope('census_loss'):
            def _ternary_transform(image):
                intensities = tf.image.rgb_to_grayscale(image) * 255
                #patches = tf.extract_image_patches( # fix rows_in is None
                #    intensities,
                #    ksizes=[1, patch_size, patch_size, 1],
                #    strides=[1, 1, 1, 1],
                #    rates=[1, 1, 1, 1],
                #    padding='SAME')
                out_channels = patch_size * patch_size
                w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
                weights =  tf.constant(w, dtype=tf.float32)
                patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')
    
                transf = patches - intensities
                transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
                return transf_norm
    
            def _hamming_distance(t1, t2):
                dist = tf.square(t1 - t2)
                dist_norm = dist / (0.1 + dist)
                dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
                return dist_sum
    
            t1 = _ternary_transform(img1)
            t2 = _ternary_transform(img2_warped)
            dist = _hamming_distance(t1, t2)
    
            transform_mask = self.create_mask(mask, [[max_distance, max_distance],
                                                [max_distance, max_distance]])
            return self.abs_robust_loss(dist, mask * transform_mask) 
 
    def compute_losses(self, batch_img1, batch_img2, batch_img3, 
            flow_fw_12, flow_bw_21, flow_fw_23, flow_bw_32,
            mask_fw_12, mask_bw_21, mask_fw_23, mask_bw_32, train=True, is_scale=True):

        img_size = get_shape(batch_img1, train=train)
        img1_warp2 = tf_warp(batch_img1, flow_bw_21['full_res'], img_size[1], img_size[2])
        img2_warp1 = tf_warp(batch_img2, flow_fw_12['full_res'], img_size[1], img_size[2])
        
        img2_warp3 = tf_warp(batch_img2, flow_bw_32['full_res'], img_size[1], img_size[2])
        img3_warp2 = tf_warp(batch_img3, flow_fw_23['full_res'], img_size[1], img_size[2])
        
        losses = {}
        
        abs_robust_mean = {}
        abs_robust_mean['no_occlusion'] = self.abs_robust_loss(batch_img1-img2_warp1, tf.ones_like(mask_fw_12)) + self.abs_robust_loss(batch_img2-img1_warp2, tf.ones_like(mask_bw_21)) + \
                                            self.abs_robust_loss(batch_img2-img3_warp2, tf.ones_like(mask_fw_23)) + self.abs_robust_loss(batch_img3-img2_warp3, tf.ones_like(mask_bw_32))
        abs_robust_mean['occlusion'] = self.abs_robust_loss(batch_img1-img2_warp1, mask_fw_12) + self.abs_robust_loss(batch_img2-img1_warp2, mask_bw_21) + \
                                            self.abs_robust_loss(batch_img2-img3_warp2, mask_fw_23) + self.abs_robust_loss(batch_img3-img2_warp3, mask_bw_32)
        losses['abs_robust_mean'] = abs_robust_mean
        
        census_loss = {}
        census_loss['no_occlusion'] = self.census_loss(batch_img1, img2_warp1, tf.ones_like(mask_fw_12), max_distance=3) + \
                    self.census_loss(batch_img2, img1_warp2, tf.ones_like(mask_bw_21), max_distance=3) + \
                    self.census_loss(batch_img2, img3_warp2, tf.ones_like(mask_fw_23), max_distance=3) + \
                    self.census_loss(batch_img3, img2_warp3, tf.ones_like(mask_bw_32), max_distance=3)
        census_loss['occlusion'] = self.census_loss(batch_img1, img2_warp1, mask_fw_12, max_distance=3) + \
                    self.census_loss(batch_img2, img1_warp2, mask_bw_21, max_distance=3) + \
                    self.census_loss(batch_img2, img3_warp2, mask_fw_23, max_distance=3) + \
                    self.census_loss(batch_img3, img2_warp3, mask_bw_32, max_distance=3)
        losses['census'] = census_loss
        
        return losses
        
    def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
        for key in keys:
            for loss_key, loss_value in losses[key].items():
                if prefix:
                    loss_name = '%s/%s/%s' % (prefix, key, loss_key)
                else:
                    loss_name = '%s/%s' % (key, loss_key)
                tf.summary.scalar(loss_name, loss_value)

    def build_no_occlusion(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        batch_img0, batch_img1, batch_img2, batch_img3, batch_img4 = iterator.get_next()
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        flow_fw_12, flow_bw_10, flow_fw_23, flow_bw_21, flow_fw_34, flow_bw_32 = pyramid_processing_five_frame(batch_img0, batch_img1, batch_img2, batch_img3, batch_img4,
            train=train, trainable=trainable, regularizer=regularizer, is_scale=is_scale)  
        


        occ_fw_12, occ_bw_21 = occlusion(flow_fw_12['full_res'], flow_bw_21['full_res'])
        mask_fw_12 = 1. - occ_fw_12
        mask_bw_21 = 1. - occ_bw_21  
        
        occ_fw_23, occ_bw_32 = occlusion(flow_fw_23['full_res'], flow_bw_32['full_res'])
        mask_fw_23 = 1. - occ_fw_23
        mask_bw_32 = 1. - occ_bw_32 


        losses = self.compute_losses(batch_img1, batch_img2, batch_img3, 
            flow_fw_12, flow_bw_21, flow_fw_23, flow_bw_32,
            mask_fw_12, mask_bw_21, mask_fw_23, mask_bw_32, train=train, is_scale=is_scale)
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)
        return losses, regularizer_loss 
    

    def build_self_supervision(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        batch_img1, batch_img2, flow_fw, flow_bw, occ_fw, occ_bw = iterator.get_next()
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        h = self.dataset_config['crop_h']
        w = self.dataset_config['crop_w']
        target_h = self.distillation_config['target_h']
        target_w = self.distillation_config['target_w']           
        offect_h = tf.random_uniform([], minval=0, maxval=h-target_h, dtype=tf.int32)
        offect_w = tf.random_uniform([], minval=0, maxval=w-target_w, dtype=tf.int32)
 
        
        batch_img1_cropped_patch = tf.image.crop_to_bounding_box(batch_img1, offect_h, offect_w, target_h, target_w)
        batch_img2_cropped_patch = tf.image.crop_to_bounding_box(batch_img2, offect_h, offect_w, target_h, target_w)     
        flow_fw_cropped_patch = tf.image.crop_to_bounding_box(flow_fw, offect_h, offect_w, target_h, target_w) 
        flow_bw_cropped_patch = tf.image.crop_to_bounding_box(flow_bw, offect_h, offect_w, target_h, target_w) 
        occ_fw_cropped_patch = tf.image.crop_to_bounding_box(occ_fw, offect_h, offect_w, target_h, target_w)
        occ_bw_cropped_patch = tf.image.crop_to_bounding_box(occ_bw, offect_h, offect_w, target_h, target_w)
        
        flow_fw_patch, flow_bw_patch = pyramid_processing_bidirection(batch_img1_cropped_patch, batch_img2_cropped_patch, 
            train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)  
        
        occ_fw_patch, occ_bw_patch = occlusion(flow_fw_patch['full_res'], flow_bw_patch['full_res'])
        mask_fw_patch = 1. - occ_fw_patch
        mask_bw_patch = 1. - occ_bw_patch

        losses = self.compute_losses(batch_img1_cropped_patch, batch_img2_cropped_patch, flow_fw_patch, flow_bw_patch, mask_fw_patch, mask_bw_patch, train=train, is_scale=is_scale)
        
        valid_mask_fw = tf.clip_by_value(occ_fw_patch - occ_fw_cropped_patch, 0., 1.)
        valid_mask_bw = tf.clip_by_value(occ_bw_patch - occ_bw_cropped_patch, 0., 1.)
        data_distillation_loss = {}
        data_distillation_loss['distillation'] = (self.abs_robust_loss(flow_fw_cropped_patch-flow_fw_patch['full_res'], valid_mask_fw) + \
                                       self.abs_robust_loss(flow_bw_cropped_patch-flow_bw_patch['full_res'], valid_mask_bw)) / 2
        losses['data_distillation'] = data_distillation_loss
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)
        return losses, regularizer_loss  


    def build(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):
        if training_mode == 'no_self_supervision':
            losses, regularizer_loss = self.build_no_occlusion(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        elif training_mode == 'self_supervision':
            losses, regularizer_loss = self.build_self_supervision(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')      
        return losses, regularizer_loss
                    
    def create_train_op(self, optim, iterator, global_step, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):  
        if self.num_gpus == 1:
            losses, regularizer_loss = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode)
            optim_loss = losses['abs_robust_mean']['no_occlusion']
            train_op = optim.minimize(optim_loss, var_list=tf.trainable_variables(), global_step=global_step)            
        else:
            tower_grads = []
            tower_losses = []
            tower_regularizer_losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_{}'.format(i)) as scope:
                            losses_, regularizer_loss_ = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode) 
                            optim_loss = losses_['census']['no_occlusion']
                            # optim_loss = losses_['abs_robust_mean']['no_occlusion']

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            grads = self.optim.compute_gradients(optim_loss, var_list=tf.trainable_variables())
                            tower_grads.append(grads)
                            tower_losses.append(losses_)
                            tower_regularizer_losses.append(regularizer_loss_)
                            #self.add_loss_summary(losses_, keys=['abs_robust_mean', 'census'], prefix='tower_%d' % i)
                                        
            grads = average_gradients(tower_grads)
            train_op = optim.apply_gradients(grads, global_step=global_step)
            
            losses = tower_losses[0].copy()
            for key in losses.keys():
                for loss_key, loss_value in losses[key].items():
                    for i in range(1, self.num_gpus):
                        losses[key][loss_key] += tower_losses[i][key][loss_key]
                    losses[key][loss_key] /= self.num_gpus
            regularizer_loss = 0.
            for i in range(self.num_gpus):
                regularizer_loss += tower_regularizer_losses[i]
            regularizer_loss /= self.num_gpus

        self.add_loss_summary(losses, keys=losses.keys())
        tf.summary.scalar('regularizer_loss', regularizer_loss)
        
        return train_op, losses, regularizer_loss

    def train(self):
        with tf.Graph().as_default(), tf.device(self.shared_device):
            self.global_step = tf.Variable(0, trainable=False)
            self.dataset, self.iterator = self.create_dataset_and_iterator(training_mode=self.training_mode)       
            self.lr_decay = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', self.lr_decay)
            self.optim = tf.train.AdamOptimizer(self.lr_decay, self.beta1)            
            self.train_op, self.losses, self.regularizer_loss = self.create_train_op(optim=self.optim, iterator=self.iterator, 
                global_step=self.global_step, regularizer_scale=self.regularizer_scale, train=True, trainable=True, is_scale=self.is_scale, training_mode=self.training_mode)
            
            merge_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir='/'.join([self.summary_dir, 'train', self.model_name]))
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
            self.saver = tf.train.Saver(var_list=self.trainable_vars + [self.global_step], max_to_keep=500)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement))
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            
            if self.is_restore_model:
                self.saver.restore(self.sess, self.restore_model)
            
            self.sess.run(tf.assign(self.global_step, 0))
            start_step = self.sess.run(self.global_step)
            self.sess.run(self.iterator.initializer)
            start_time = time.time()
            for step in range(start_step+1, self.iter_steps+1):
                _, census_no_occlusion, census_occlusion = self.sess.run([self.train_op,
                    self.losses['census']['no_occlusion'], self.losses['census']['occlusion']])
                if np.mod(step, self.display_log_interval) == 0:
                    print('step: %d time: %.6fs, census_no_occlusion: %.6f, census_occlusion: %.6f' % 
                        (step, time.time() - start_time, census_no_occlusion, census_occlusion))
                
                if np.mod(step, self.write_summary_interval) == 0:
                    summary_str = self.sess.run(merge_summary)
                    summary_writer.add_summary(summary_str, global_step=step)
                
                if np.mod(step, self.save_checkpoint_interval) == 0:
                    self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                    write_meta_graph=False, write_state=False) 
    
                    
    def test(self, restore_model, save_dir, is_normalize_img=True):
        from test_datasets import BasicDataset
        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'], is_normalize_img=is_normalize_img)
        save_name_list = dataset.data_list[:, -1]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img0, batch_img1, batch_img2 = iterator.get_next()
        img_shape = tf.shape(batch_img0)
        h = img_shape[1]
        w = img_shape[2]
        
        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
        
        batch_img0 = tf.image.resize_images(batch_img0, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(batch_img1, [new_h, new_w], method=1, align_corners=True)
        batch_img2 = tf.image.resize_images(batch_img2, [new_h, new_w], method=1, align_corners=True)
        
        flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
        flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
        flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)
        
        flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
        flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = sess.run([flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color])
            misc.imsave('%s/flow_fw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_fw_color[0])
            misc.imsave('%s/flow_bw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_bw_color[0])
            write_flo('%s/flow_fw_%s.flo' % (save_dir, save_name_list[i]), np_flow_fw[0])
            write_flo('%s/flow_bw_%s.flo' % (save_dir, save_name_list[i]), np_flow_bw[0])
            print('Finish %d/%d' % (i+1, dataset.data_num))

    def generate_fake_flow_occlusion(self, restore_model, save_dir):
        from test_datasets import BasicDataset
        data_list_file = "./dataset/KITTI/id2.txt"
        img_dir = "../datasets/KITTI/training"
        save_dir = "../datasets/KITTI_flow/training"
        dataset = BasicDataset(data_list_file=data_list_file, img_dir=img_dir, is_normalize_img=False)
        save_name_list = dataset.data_list[:, 3]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img0, batch_img1, batch_img2 = iterator.get_next()
        img_shape = tf.shape(batch_img0)
        h = img_shape[1]
        w = img_shape[2]
        
        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
        
        batch_img0 = tf.image.resize_images(batch_img0, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(batch_img1, [new_h, new_w], method=1, align_corners=True)
        batch_img2 = tf.image.resize_images(batch_img2, [new_h, new_w], method=1, align_corners=True)
        
        flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
        flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
        flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)
        
        flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
        flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            np_flow_fw, np_flow_bw = sess.run([flow_fw['full_res'], flow_bw['full_res']])
            #misc.imsave('%s/flow_fw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_fw_color[0])
            #misc.imsave('%s/flow_bw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_bw_color[0])
            write_flo('%s/flow_fw_%s.flo' % (save_dir, save_name_list[i]), np_flow_fw[0])
            write_flo('%s/flow_bw_%s.flo' % (save_dir, save_name_list[i]), np_flow_bw[0])
            print('Finish %d/%d' % (i+1, dataset.data_num))

            #todo: name saving properly according to image
            #segments = slic(img) and save
        

        
        
    

        
            
              
