
"""
File to compute local thickness.

Author: L. L. Longren
"""


import numpy as np
import matplotlib.pyplot as plt
import time

t1 = time.time()

## Create a test image
sl = int(2**6)
I = np.zeros((sl, sl, sl))
I[sl//8:-sl//8, sl//8:-sl//8, sl//8:-sl//8] = 1
I[sl//4:-sl//4, sl//4:-sl//4, sl//4:-sl//4] = 0
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
from Local_Thickness_translation import Local_Thickness_Parallel, Clean_Up_Local_Thickness

LTP = Local_Thickness_Parallel(distanceridgeArray)
LTP_out = LTP.run()

CULT = Clean_Up_Local_Thickness(LTP_out)
CULT_out = CULT.run()

# Mask the local thickness map
result = np.where(I, CULT_out, np.zeros(np.shape(I)))

t2 = time.time()
print('Total time: {:.1f} s'.format(t2 - t1))

# Visualize

fig, axs = plt.subplots(ncols=4, dpi=200)
axs[0].imshow(I[sl//2], cmap='gray')
axs[1].imshow(LTP_out[sl//2], cmap='viridis')
axs[2].imshow(CULT_out[sl//2], cmap='viridis')
axs[3].imshow(result[sl//2], cmap='viridis')
axs[0].set_title('input')
axs[1].set_title('LTP')
axs[2].set_title('CULT')
axs[3].set_title('result')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[3].axis('off')
plt.show()

