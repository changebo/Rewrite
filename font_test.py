import numpy as np
import matplotlib.image as mplimg
from matplotlib import cm

def read_font_data(font, unit_scale):
    F = np.load(font)
    if unit_scale:
        return F / 255.
    return F

font_test = read_font_data("SentyCHALKoriginal.npy",1)

char_num = font_test.shape[0]
for i in range(char_num):
    mplimg.imsave(str(i)+'.png',np.uint8(font_test[i,:,:]), cmap = cm.gray)
