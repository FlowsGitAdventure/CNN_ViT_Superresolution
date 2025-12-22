import gzip
import shutil
import os


def compress_file(input_path):
    """
    Compresses a file to .gz using streaming (low RAM usage).
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # Determine output path (add .gz)
    output_path = input_path + ".gz"

    print(f"Compressing: {input_path}")
    print(f"Target:      {output_path}")

    try:
        # Open source file in binary read mode
        with open(input_path, 'rb') as f_in:
            # Open destination file in gzip binary write mode
            with gzip.open(output_path, 'wb') as f_out:
                # Copy data in 64MB chunks to avoid loading full file into RAM
                shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)

        print("Success! Compression complete.")

        # Optional: Delete the original .nii to free up space
        # os.remove(input_path)

    except Exception as e:
        print(f"Compression failed: {e}")


if __name__ == "__main__":
    # REPLACE THIS with the path to your .nii file
    target_file = "output_hr.nii"

    compress_file(target_file)