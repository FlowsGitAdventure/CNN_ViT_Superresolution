#import torch
#from UNetSuperRes import UNet  # Import our modified UNet

#input_image = torch.rand((1, 3, 32, 32, 32))
#
## Initialize the model with 3 input channels and 3 output channels (RGB)
#model = UNet(in_channels=3, out_channels=3)
#
#output = model(input_image)
#
#print(f"Input tensor size: {input_image.size()}")
#print(f"Output tensor size: {output.size()}")
#
## Expected output: torch.Size([1, 3, 512, 512])
#assert output.size() == torch.Size([1, 3, 128, 128, 128])
#print("\nSuccess! Output size matches target HR size.")


import torch
from UNetSuperRes import UNet


batch_size = 1
channels = 1
time_points = 10
depth, height, width = 32, 32, 32 # Low-Res Input spatial dims

input_fmri = torch.rand((batch_size, channels, time_points, depth, height, width))
input_model = input_fmri.view(batch_size, channels * time_points, depth, height, width)

print(f"Original Shape: {input_fmri.shape}")
print(f"Model Input Shape: {input_model.shape}")
model = UNet(in_channels=channels * time_points,
             out_channels=channels * time_points)
output_model = model(input_model)
out_depth, out_height, out_width = output_model.shape[2:]
output_fmri = output_model.view(batch_size, channels, time_points, out_depth, out_height, out_width)

print(f"Model Output Shape: {output_model.shape}")
print(f"Final fMRI Shape:   {output_fmri.shape}")