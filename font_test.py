import numpy as np
import matplotlib.image as mplimg
from matplotlib import cm

font_test = np.load("simhei.npy")
font_test = font_test/255.

char_num = font_test.shape[0]
for i in range(char_num):
    mplimg.imsave(str(i)+'.jpg',np.uint8(font_test[i,:,:]), cmap = cm.gray)
