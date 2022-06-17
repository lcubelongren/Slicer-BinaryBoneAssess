
"""
File to compute local thickness.

Author: L. L. Longren
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time

t1 = time.time()

## Create a test image
sl = int(2**6)
def hollowSquare(sl):
    I = np.zeros((sl, sl, sl), dtype='uint8')
    I[sl//8:-sl//8, sl//8:-sl//8, sl//8:-sl//8] = 1
    I[sl//4:-sl//4, sl//4:-sl//4, sl//4:-sl//4] = 0
    return I
I = hollowSquare(sl)
io.imsave('output/hollowSquare_sl-{}.tif'.format(sl), I)

#I = io.imread('/media/lukel/LL External/testdata/mouse_skull_reference/Data/4074_skulltif.tif').astype('uint8')
#I = I[:sl, :sl, :sl]

print('Input shape: {}'.format(I.shape))

# Calculations from pre-existing libraries
from skimage import morphology
from scipy import ndimage

print('Computing images...')
distanceArray = ndimage.distance_transform_edt(I)
skeletonArray = morphology.skeletonize_3d(I)

print('Combining images...')
very_small_value = 1e-9
medialaxisArray = np.where(skeletonArray, distanceArray, np.zeros(np.shape(distanceArray)))
distanceridgeArray = np.where(medialaxisArray == 0, I * very_small_value, medialaxisArray)

# Calculations using the translation from Java
from Local_Thickness import Local_Thickness_Parallel, Clean_Up_Local_Thickness

LTP = Local_Thickness_Parallel(distanceridgeArray)
LTP_out = LTP.run()

CULT = Clean_Up_Local_Thickness(LTP_out)
CULT_out = CULT.run()

# Mask the local thickness map
result = np.where(I, CULT_out, np.zeros(np.shape(I)))
io.imsave('output/pythonOut/hollowSquare_sl-{}_LocThk.tif'.format(sl), result)

t2 = time.time()
print('Total time: {:.1f} s'.format(t2 - t1))

# Visualize

def viz():
    nrows, ncols = 3, 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=200)
    for row in range(nrows):
        slice_num = (row+1)*(sl//4)
        axs[row,0].imshow(I[slice_num], cmap='gray')
        axs[row,1].imshow(LTP_out[slice_num], cmap='viridis')
        axs[row,2].imshow(CULT_out[slice_num], cmap='viridis')
        axs[row,3].imshow(result[slice_num], cmap='viridis')
        axs[row,0].set_ylabel('{}/{}'.format(slice_num, sl))
        for col in range(ncols): 
            axs[row,col].set_xticks([]), axs[row,col].set_yticks([])
        if row == 0:
            axs[row,0].set_title('input')
            axs[row,1].set_title('LTP')
            axs[row,2].set_title('CULT')
            axs[row,3].set_title('result')
    plt.show()
# viz()

