# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from PIL import Image

a = np.array(Image.open('01.jpg'))
#三维数组变换
b = 255*(a/255)**2
#生成图像对象
im = Image.fromarray(b.astype('uint8'))
#保存图像
im.save('C:\\Users\\baochun_kan\\Desktop\\01.jpg')