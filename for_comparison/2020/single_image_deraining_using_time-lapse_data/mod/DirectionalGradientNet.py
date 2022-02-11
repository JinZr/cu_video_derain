import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DirectionalGradientNet(nn.Module):

    def __init__(self, k: int, device) -> None:
        super(DirectionalGradientNet, self).__init__()

        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(
            0).unsqueeze(0).to(device=device)
        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(
            0).unsqueeze(0).to(device=device)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        self.soft_assignment = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=k,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.Softmax()
        )

    def forward(self, s):
        grad_x = F.conv2d(s, self.weight_x, padding=1)
        grad_y = F.conv2d(s, self.weight_y, padding=1)
        theta_i = torch.atan(grad_x / grad_y)
        alpha_k_theta_i = self.soft_assignment(theta_i)
        return theta_i, alpha_k_theta_i


if __name__ == '__main__':
    import numpy as np
    module = DirectionalGradientNet(k=4, device=torch.device(
        'cpu' if not torch.cuda.is_available() else 'cuda:0')).float()
    module(
        torch.from_numpy(np.random.rand(4, 1, 128, 128)).float()
    )
