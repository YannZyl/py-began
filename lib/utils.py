# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import os
import numpy as np

def create_dir():
    dirs = ['data/checkpoint', 'data/output', 'data/interpolate']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def save_images(images, out_dir, im_name, num_per_rows=8):
    n, h, w, c = images.shape
    rows = int(np.ceil(n/num_per_rows))
    # fill image into grid
    images = (images + 1.0) * 127.5
    images = np.clip(np.array(images), 0, 255).astype(np.uint8)
    recon_images = np.zeros([rows*h, num_per_rows*w, c], dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // num_per_rows
        c = idx %  num_per_rows
        recon_images[r*h:(r+1)*h, c*w:(c+1)*w, :] = image 
    recon_images = cv2.cvtColor(recon_images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir,'{}.jpg'.format(im_name)), recon_images)
    
def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
