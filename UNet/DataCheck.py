import numpy
import nibabel as nib
import os


def get_file_names(file_paths):
    dir_list = os.listdir(file_paths)
    dir_list = [f for f in dir_list if '.nii' in f]
    return dir_list


def check_data_uniformity(path):
    if not path:
        print("Error: File path list is empty.")
        return False, None, []
    files = get_file_names(path)
    try:
        filepath = os.path.join(path, files[0])
        img_4d = nib.load(filepath)
        first_volume = img_4d.get_fdata()
        expected_shape = first_volume.shape
    except Exception as e:
        print(f"Error laoding first volume {files[0]}: {e}")
        return False, None, []

    misaligned_files = []

    for i, file_path in enumerate(files[1:]):
        try:
            filepath = os.path.join(path, files[0])
            current_img_4d = nib.load(filepath)
            current_volume = current_img_4d.get_fdata()

            if current_volume.shape != expected_shape:
                misaligned_files.append(file_path)
                print(f"Mismatch found: File '{file_path}' has shape {current_volume.shape}, expected {expected_shape}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            misaligned_files.append(file_path)

        if not misaligned_files:
            print(f"Success! All {len(files)} volumes have the consistent shape {expected_shape}.")
            return True, expected_shape, []
        else:
            print(f"Failure! {len(misaligned_files)} files have mismatched shapes.")
            return False, expected_shape, misaligned_files


if __name__ == "__main__":
    check_data_uniformity('../Downsampling/Low_Res_08_Rician')
    check_data_uniformity('../Downsampling/High_Res_08_Rician')
