# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from inputs import data_loader
from tensorflow.contrib import slim

class BEGAN:
    def __init__(self, args):
        self.x_dim = args.image_scale
        self.z_dim = args.noise_scale
        self.hidden_num = args.hidden_num
        self.repeat_num = int(np.log2(self.x_dim)-2)
        self.batch_size = args.batch_size
        
	self.beta1 = args.beta1
	self.beta2 = args.beta2
        self.gamma = args.gamma
        self.lambda_k = args.lambda_k
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr
        self.i_lr = args.i_lr
        self.d_lr_low_boundary = args.d_lr_low_boundary
        self.g_lr_low_boundary = args.g_lr_low_boundary
        
        self.data_dir = args.data_dir
        # build training model
        self.build_train_model()
        # build interpolate model
        self.build_interp_model()
        # prepare session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.prepare_session()
    
    def build_train_model(self):
        # input and noise 
        with tf.name_scope('Data_flow'):
            self.x = data_loader(self.data_dir, self.batch_size, self.x_dim)
            self.z = tf.random_uniform([self.batch_size,self.z_dim], minval=-1.0, maxval=1.0, name='noise')           
        # reconstruction/output of G and D in the training step
        with tf.variable_scope('Outputs'):
            G_, var_G = self.generator(self.z)
            D_, var_D = self.discriminator(G_)
            D , _ = self.discriminator(self.x, reuse=True)
            self.S = G_
        # build D loss function
        with tf.name_scope('Loss_D'):
            self.k_t = tf.Variable(0., trainable=False, name='k_t')
            d_loss_real = tf.reduce_mean(tf.abs(D-self.x))
            d_loss_fake = tf.reduce_mean(tf.abs(D_-G_))
            self.d_loss = d_loss_real - self.k_t*d_loss_fake
        # build G loss function
        with tf.name_scope('Loss_G'):
            self.g_loss = tf.reduce_mean(tf.abs(D_-G_))
        # D optimizer
        with tf.name_scope('Optim_D'):
            d_lr = tf.Variable(self.d_lr, trainable=False, name='d_lr')
            d_optimizer = tf.train.AdamOptimizer(d_lr).minimize(self.d_loss, var_list=var_D)
        # G optimizer
        with tf.name_scope('Optim_G'):
            g_lr = tf.Variable(self.g_lr, trainable=False, name='g_lr')
            g_optimizer = tf.train.AdamOptimizer(g_lr).minimize(self.g_loss, var_list=var_G)
        # Train measure item
        with tf.name_scope('Measure'):
            balance = self.gamma * d_loss_real - self.g_loss
            self.measure = d_loss_real + tf.abs(balance)
        # update items, including learning rate update and kt update 
        with tf.name_scope('Update'):
            with tf.control_dependencies([d_optimizer, g_optimizer]):
                self.k_t_update =  tf.assign(self.k_t, tf.clip_by_value(self.k_t+self.lambda_k*balance, 0, 1), name='kt_update')

        self.d_lr_update = tf.assign(d_lr, tf.maximum(d_lr*0.5, self.d_lr_low_boundary), name='d_lr_update')
        self.g_lr_update = tf.assign(g_lr, tf.maximum(g_lr*0.5, self.g_lr_low_boundary), name='g_lr_update')
            
    def build_interp_model(self):
        # feed dict, couple image and noize
        with tf.name_scope('Data_couple'):
            self.cz = tf.Variable(tf.zeros([self.batch_size,self.z_dim]), name='interp_noise')
        # reconstruction/output of G in the interpolating step
        with tf.variable_scope('Outputs'):
            cs, _ = self.generator(self.cz, reuse=True)
        # build G loss function 
        with tf.name_scope('Loss_I'):
            self.i_loss = tf.reduce_mean(tf.abs(self.x-cs))
        # I optimizer
        with tf.name_scope('Optim_I'):
            i_optimizer = tf.train.AdamOptimizer(self.i_lr).minimize(self.i_loss, var_list=[self.cz])
        # clip noise into [-1,1]
        with tf.name_scope('Noise_clip'):
            with tf.control_dependencies([i_optimizer]):
                self.clip_noise = tf.assign(self.cz, tf.clip_by_value(self.cz, -1.0, 1.0))

    def prepare_session(self):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='data/checkpoint')
        if ckpt is not None:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            tf.get_variable_scope().reuse_variables()
        else:
            self.session.run(tf.global_variables_initializer())
    
    def save_session(self, global_step):
        self.saver.save(self.session, 'data/checkpoint/began.ckpt', global_step=global_step)
        
    def generator(self, z, reuse=None):
        with tf.variable_scope('G', reuse=reuse) as vs:
            # construct to a conv map with size: 8x8xhidden_num
            num_outputs = int(np.prod([8,8,self.hidden_num]))
            x = slim.fully_connected(z, num_outputs, activation_fn=None)
            x = tf.reshape(x, [-1,8,8,self.hidden_num])

            for idx in range(self.repeat_num):
                x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    _, h, w, _ = x.get_shape().as_list()
                    x = tf.image.resize_nearest_neighbor(x, (h*2,w*2))

            x = slim.conv2d(x, 3, 3, 1, activation_fn=None)
        
        variables = tf.contrib.framework.get_variables(vs)
        return x, variables
    
    def discriminator(self, x, reuse=None):
        with tf.variable_scope('D', reuse=reuse) as vs:
            x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
            
            num_channels = self.hidden_num
            for idx in range(self.repeat_num):
                num_channels = (idx+1) * self.hidden_num
                x = slim.conv2d(x, num_channels, 3, 1, activation_fn=tf.nn.elu)
                x = slim.conv2d(x, num_channels, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    x = slim.conv2d(x, num_channels, 3, 2, activation_fn=tf.nn.elu)
            
            x = tf.reshape(x, [-1,8*8*num_channels])
            x = slim.fully_connected(x, self.z_dim, activation_fn=None)
            
            num_outputs = int(np.prod([8,8,self.hidden_num]))
            x = slim.fully_connected(x, num_outputs, activation_fn=None)
            x = tf.reshape(x, [-1,8,8,self.hidden_num])
            
            for idx in range(self.repeat_num):
                x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < self.repeat_num - 1:
                    _, h, w, _ = x.get_shape().as_list()
                    x = tf.image.resize_nearest_neighbor(x, (h*2,w*2))
            
            x = slim.conv2d(x, 3, 3, 1, activation_fn=None)
        
        variables = tf.contrib.framework.get_variables(vs)
        return x, variables
