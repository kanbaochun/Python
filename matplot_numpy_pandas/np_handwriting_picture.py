# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from PIL import Image

a = np.asanyarray(Image.open('C:\\Program Files\\Python36\\01.jpg').convert('L')).astype('float')

depth = 10.
grad = np.gradient(a)  #取图像灰度的梯度值
grad_x ,grad_y = grad  #分别取横纵坐标的梯度值
grad_x = grad_x*depth/100.
grad_y = grad_y*depth/100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x/A
uni_y = grad_y/A
uni_z = 1./A

vec_e1 = np.pi/2.2
vec_az = np.pi/4.
dx = np.cos(vec_e1)*np.cos(vec_az)
dy = np.cos(vec_e1)*np.sin(vec_az)
dz = np.sin(vec_e1)

b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)
b = b.clip(0,255)

im = Image.fromarray(b.astype('uint8'))  #重构图像
im.save('C:\\Users\\baochun_kan\\Desktop\\01.jpg')