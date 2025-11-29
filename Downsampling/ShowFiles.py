import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

#img = nib.load("C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func\\sub-04_ses-r08_task-coverage_rec-dico_bold.nii.gz")
img = nib.load("C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\Downsampling\\High_Res_08_Rician\\sub-04_ses-r08_task-coverage_bold_HR.nii.gz")
# img = nib.load("./output_hr.nii.gz")
data = img.get_fdata()
plt.imshow(data[:, :, 15, 0], cmap='gray')
plt.show()

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_slice(file_path, timepoint=0, slice_idx=None, axis=2):
    """
    Visualizes a single slice from a large 4D NIfTI file without loading the whole file.
    axis: 0=Sagittal, 1=Coronal, 2=Axial
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    print(f"Loading header for: {os.path.basename(file_path)}")
    img = nib.load(file_path)
    shape = img.shape
    print(f"Dimensions: {shape}")

    # Default to middle slice if not specified
    if slice_idx is None:
        slice_idx = shape[axis] // 2

    print(f"Extracting Slice: Axis={axis}, Index={slice_idx}, Time={timepoint}...")

    # --- MEMORY SAFE SLICING ---
    # We use img.dataobj[...] instead of img.get_fdata()
    # This reads specific bytes directly from the hard drive.

    try:
        if len(shape) == 4:
            # 4D Data: (X, Y, Z, T)
            if axis == 0:
                # Slice X (Sagittal)
                slice_data = img.dataobj[slice_idx, :, :, timepoint]
            elif axis == 1:
                # Slice Y (Coronal)
                slice_data = img.dataobj[:, slice_idx, :, timepoint]
            else:
                # Slice Z (Axial) - This matches your data[:, :, slice, 0] example
                slice_data = img.dataobj[:, :, slice_idx, timepoint]
        else:
            # 3D Data: (X, Y, Z)
            if axis == 0:
                slice_data = img.dataobj[slice_idx, :, :]
            elif axis == 1:
                slice_data = img.dataobj[:, slice_idx, :]
            else:
                slice_data = img.dataobj[:, :, slice_idx]

        # Convert ONLY this small 2D slice to numpy float32
        slice_data = np.array(slice_data, dtype=np.float32)

        # Plotting
        plt.figure(figsize=(10, 8))
        # rotate 90 degrees usually makes MRI look "upright" in Matplotlib
        plt.imshow(np.rot90(slice_data), cmap='gray')
        plt.title(f"Slice {slice_idx} (Axis {axis}) @ Time {timepoint}\nShape: {slice_data.shape}")
        plt.colorbar(label='Intensity')
        plt.axis('off')
        plt.show()

    except IndexError:
        print(f"Error: Slice index {slice_idx} or Timepoint {timepoint} is out of bounds for shape {shape}")


# if __name__ == "__main__":
#     # --- CHANGE PATH HERE ---
#     #file_path = "./data/LR/sub-01_task-rest.nii.gz"
#     file_path = "./output_hr.nii" # Use this to check your result
#
#     # Visualize Middle Axial Slice at Timepoint 0
#     visualize_slice(file_path, timepoint=0, axis=2, slice_idx=15)
