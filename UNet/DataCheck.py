# import numpy
# import nibabel as nib
# import os
#
#
# def get_file_names(file_paths):
#     dir_list = os.listdir(file_paths)
#     dir_list = [f for f in dir_list if '.nii' in f]
#     return dir_list
#
#
# def check_data_uniformity(path):
#     if not path:
#         print("Error: File path list is empty.")
#         return False, None, []
#     files = get_file_names(path)
#     try:
#         filepath = os.path.join(path, files[0])
#         img_4d = nib.load(filepath)
#         first_volume = img_4d.get_fdata()
#         expected_shape = first_volume.shape
#     except Exception as e:
#         print(f"Error laoding first volume {files[0]}: {e}")
#         return False, None, []
#
#     misaligned_files = []
#
#     for i, file_path in enumerate(files[1:]):
#         try:
#             filepath = os.path.join(path, files[0])
#             current_img_4d = nib.load(filepath)
#             current_volume = current_img_4d.get_fdata()
#
#             if current_volume.shape != expected_shape:
#                 misaligned_files.append(file_path)
#                 print(f"Mismatch found: File '{file_path}' has shape {current_volume.shape}, expected {expected_shape}")
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")
#             misaligned_files.append(file_path)
#
#         if not misaligned_files:
#             print(f"Success! All {len(files)} volumes have the consistent shape {expected_shape}.")
#             return True, expected_shape, []
#         else:
#             print(f"Failure! {len(misaligned_files)} files have mismatched shapes.")
#             return False, expected_shape, misaligned_files
#
#
# if __name__ == "__main__":
#     check_data_uniformity('../Downsampling/Low_Res_08_Rician')
#     check_data_uniformity('../Downsampling/High_Res_08_Rician')

import os
import nibabel as nib

# --- CONFIGURATION ---
LR_DIR = "../Downsampling/Low_Res_08_Rician_test"
HR_DIR = "../Downsampling/High_Res_08_Rician_test"

# The majority shapes you want to KEEP
TARGET_SHAPE_LR = (104, 80, 16)
TARGET_SHAPE_HR = (208, 160, 32)


# ---------------------

def clean_dataset(directory, target_shape):
    print(f"Scanning {directory}...")
    print(f"Targeting to KEEP shape: {target_shape}")

    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    deleted_count = 0
    kept_count = 0

    for f in files:
        file_path = os.path.join(directory, f)

        try:
            # Load only the header to be fast (doesn't load the full image data)
            img = nib.load(file_path)

            # Check dimensions
            # Note: nibabel shapes are typically (X, Y, Z).
            # Make sure this matches your print output order.
            if img.shape != target_shape:
                print(f"  [DELETE] {f} | Found shape {img.shape} != Target {target_shape}")

                # --- THE DELETE COMMAND ---
                img.uncache()  # Release file handle before deleting (windows specific fix)
                os.remove(file_path)
                # --------------------------

                deleted_count += 1
            else:
                kept_count += 1

        except Exception as e:
            print(f"  [ERROR] Could not process {f}: {e}")

    print(f"Finished {directory}.")
    print(f"  Deleted: {deleted_count}")
    print(f"  Kept:    {kept_count}\n")


if __name__ == "__main__":
    # Double check paths exist before running
    if os.path.exists(LR_DIR) and os.path.exists(HR_DIR):
        clean_dataset(LR_DIR, TARGET_SHAPE_LR)
        clean_dataset(HR_DIR, TARGET_SHAPE_HR)
        print("Cleanup complete. Dataset is now uniform.")
    else:
        print("Error: Check your directory paths.")