import torch
import torch.nn as nn
import torch.nn.functional as f


class SobelEdgeLoss3D(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Standard 3D sobel filter
        kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                  [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)

        self.kernel_x = kernel_x.unsqueeze(0).to(device, non_blocking=True)
        self.kernel_y = self.kernel_x.transpose(2, 3)
        self.kernel_z = self.kernel_x.transpose(2, 4)

    def forward(self, pred, target):
        channels = pred.shape[1]
        kx = self.kernel_x.repeat(channels, 1, 1, 1, 1)
        ky = self.kernel_y.repeat(channels, 1, 1, 1, 1)
        kz = self.kernel_z.repeat(channels, 1, 1, 1, 1)

        grad_x_p = f.conv3d(pred, kx, padding=1, groups=channels)
        grad_y_p = f.conv3d(pred, ky, padding=1, groups=channels)
        grad_z_p = f.conv3d(pred, kz, padding=1, groups=channels)

        grad_x_t = f.conv3d(target, kx, padding=1, groups=channels)
        grad_y_t = f.conv3d(target, ky, padding=1, groups=channels)
        grad_z_t = f.conv3d(target, kz, padding=1, groups=channels)

        # Penalize the difference between the edge maps
        edge_loss = f.l1_loss(grad_x_p, grad_x_t) + f.l1_loss(grad_y_p, grad_y_t) + f.l1_loss(grad_z_p, grad_z_t)
        return edge_loss
