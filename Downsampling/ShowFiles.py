import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

#img = nib.load("C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func\\sub-04_ses-r08_task-coverage_rec-dico_bold.nii.gz")
img = nib.load("C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\Downsampling\\DownscaledData\\LR_generated.nii.gz")
data = img.get_fdata()

plt.imshow(data[:, :, 15, 0], cmap='gray')
plt.show()

