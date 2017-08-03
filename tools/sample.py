# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import trange
from lib.began import BEGAN
from lib.utils import slerp, save_images, create_dir

class Sampler:
    def __init__(self, args):
        self.mode = args.mode
        self.image1 = args.image1
        self.image2 = args.image2
        self.interp_dir = args.interp_dir
        self.interp_step = args.interp_step
	create_dir()
        self.net = BEGAN(args)
        
    def file_check(self):
        if self.mode == 3:
            return True
        if self.image1 is None:
            print 'Image1 has not entered.'
            return False
        if not os.path.exists(self.image1):
            print 'Image1: {} not exist.'.format(self.image1)
            return False
        if self.mode == 1 and self.image2 is None:
            print 'Image2 has not entered.'
            return False
        if self.mode == 1 and not os.path.exists(self.image2):
            print 'image2: {} not exist.'.format(self.image2)
            return False
        return True
    
    
    def interpolate(self): 
        result = self.file_check()
        if not result:
            return
        n, h, w, c = self.net.x.get_shape().as_list() 
        if self.mode == 1:
            image_1 = cv2.imread(self.image1)
            image_1 = cv2.resize(image_1, (self.net.x_dim,self.net.x_dim))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_2 = np.flip(image_1, 1)
            batch_image = np.array([image_1,image_2])/127.5-1.0
        elif self.mode == 2:
            image_1 = cv2.imread(self.image1)
            image_1 = cv2.resize(image_1, (self.net.x_dim,self.net.x_dim))
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            image_2 = cv2.imread(self.image2)
            image_2 = cv2.resize(image_2, (self.net.x_dim,self.net.x_dim))
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
            batch_image = np.array([image_1,image_2])/127.5-1.0
        else:
            tf.train.start_queue_runners(self.net.session)
            batch_image = self.net.session.run(self.net.x)
            batch_image = batch_image[0:2,...]
        # bp map real image to noize
        for step in trange(self.interp_step, desc='Training progress'):
            i_loss, _ = self.net.session.run([self.net.i_loss,self.net.clip_noise], {self.net.cx: batch_image})
            print("[{}/{}] Loss_I: {:.6f}".format(step, self.interp_step, i_loss))
        
        noise = self.net.session.run(self.net.cz, feed_dict={self.net.cx: batch_image})
        #noise1, noise2 = self.net.session.run(self.net.cz)
        noise1, noise2 = np.split(noise, 2, 0)
        noise1 = np.squeeze(noise1)
        noise2 = np.squeeze(noise2)
        # interplate between noise
        decodes = []
        decodes.append(batch_image[0])
        print noise1.shape, noise1
        print noise2.shape, noise2
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(noise1, noise2)])
	    z = np.expand_dims(z,0)
            z_double = np.concatenate([z]*self.net.batch_size, 0)
            im = self.net.session.run(self.net.S, {self.net.z: z_double})
            #im = np.split(im, 2, 0)[0]
            decodes.append(im[0])
        decodes.append(batch_image[1])
        #decodes = np.array(decodes).transpose(1,0,2,3,4)
        #decodes = np.reshape(decodes, [-1,h,w,c])
        decodes = np.array(decodes)
        # save images
        im_name = 'interpolate'
        save_images(decodes, self.interp_dir, im_name, num_per_rows=12)       
