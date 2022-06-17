
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


sl = int(2**6)
# Compare to Fiji
result_self = io.imread('output/pythonOut/hollowSquare_sl-{}_LocThk.tif'.format(sl))
result_IJ = io.imread('output/fijiOut/hollowSquare_sl-{}_LocThk.tif'.format(sl))

incorrect = (result_self - result_IJ) != 0

plt.imshow(incorrect[sl//2])
plt.show()

