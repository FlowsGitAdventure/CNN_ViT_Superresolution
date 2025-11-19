import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np

img_original = cv2.imread('BildKohlerF.png', 0)

[m, n] = img_original.shape
print(f'Image shape: {m}, {n}')

factor = 4
img2 = np.zeros((m//factor, n//factor), dtype=np.int)
for i in range(0, m, factor):
    for j in range(0, n, factor):
        try:

            img2[i//factor][j//factor] = img_original[i][j]
        except IndexError:
            pass

plt.savefig('down.png')

