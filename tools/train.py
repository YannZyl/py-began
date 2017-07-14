# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import trange
from lib.began import BEGAN
from lib.utils import save_images, create_dir

class Trainer:
    def __init__(self, args):
        self.max_step = args.max_step
        self.log_step = args.log_step
        self.save_step = args.save_step
        self.lr_update_step = args.lr_update_step
        self.out_dir = args.out_dir
	create_dir()
        self.net = BEGAN(args)
    
    def train(self):
        z_fixed = np.random.uniform(-1., 1., [self.net.batch_size,self.net.z_dim])
        # define queue runner and coordinator
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=self.net.session, coord=coord)
        # training 
        for step in trange(self.max_step, desc='Training progress'):
            fetch_dict = {
                'update_kt': self.net.k_t_update,
                'measure':   self.net.measure
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    'g_loss': self.net.g_loss,     
                    'd_loss': self.net.d_loss,
                    'k_t':    self.net.k_t
                })
           
            # run graph
            result = self.net.session.run(fetch_dict)
            measure = result['measure']
            
            # print log
            if step % self.log_step == 0:
                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))
            
            # sample/test internal
            if step % (self.log_step * 10) == 0:
                self.generate(z_fixed, step)
            
            # update learning rate
            if step % self.lr_update_step == self.lr_update_step - 1:
                self.net.session.run([self.net.g_lr_update, self.net.d_lr_update])
            
            # save session
            if step % self.save_step == 0 and step != 0:
                self.net.save_session(step)
        coord.request_stop()
        coord.join(thread)
    
    def generate(self, noise, global_step):
        # generate images
        images = self.net.session.run(self.net.S, {self.net.z: noise})
        # save images into disk
        im_name = 'generate_{}'.format(global_step)
        save_images(images, self.out_dir, im_name, num_per_rows=4)
