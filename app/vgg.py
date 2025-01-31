"""VGG Classes and methods."""

import torch


class VGGALrn(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG A Model."""
        super(VGGALrn, self).__init__()

        self.a_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        # paper cites "ImageNet Classification with Deep Convolutional Neural Networks"
        # which uses this config for LRN
        self.a_lrn = torch.nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_2 = torch.nn.Conv2d(64, 128, (3, 3), strid=1, padding=1)
        self.a_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.a_4 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_5 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_6 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_7 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_8 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

        self.conv_net = torch.nn.Sequential(
            self.a_1,
            self.a_mx_1,
            self.a_2,
            self.a_mx_2,
            self.a_3,
            self.a_4,
            self.a_mx_3,
            self.a_5,
            self.a_6,
            self.a_mx_4,
            self.a_7,
            self.a_8,
            self.a_mx_5,
        )

        self.fc_net = torch.nn.Sequential(
            self.fc1, self.re1, self.fc2, self.re2, self.fc3, self.re3
        )

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""
        pass


class VGGA(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG A-lrn Model."""
        super(VGGA, self).__init__()

        self.a_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_2 = torch.nn.Conv2d(64, 128, (3, 3), strid=1, padding=1)
        self.a_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.a_4 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_5 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_6 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.a_7 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_8 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

        self.conv_net = torch.nn.Sequential(
            self.a_1,
            self.a_mx_1,
            self.a_2,
            self.a_mx_2,
            self.a_3,
            self.a_4,
            self.a_mx_3,
            self.a_5,
            self.a_6,
            self.a_mx_4,
            self.a_7,
            self.a_8,
            self.a_mx_5,
        )

        self.fc_net = torch.nn.Sequential(
            self.fc1, self.re1, self.fc2, self.re2, self.fc3, self.re3
        )

    def init_params(self):
        """Initialize model parameters."""
        pass

    def forward(self, input: torch.Tensor):
        """Model Forward Pass."""
        output = self.conv_net(input)
        output = self.fc_net(output)
        output = self.soft(output)

        return output


class VGGB(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG B Model."""
        super(VGGB, self).__init__()

        self.b_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.b_2 = torch.nn.Conv2d(64, 64, (3, 3), strid=1, padding=1)
        self.b_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.b_3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.b_4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.b_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.b_5 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.b_6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.b_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.b_7 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.b_8 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.b_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.b_9 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.b_10 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.a_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""
        pass


class VGGC(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG C Model."""
        super(VGGC, self).__init__()

        self.c_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.c_2 = torch.nn.Conv2d(64, 64, (3, 3), strid=1, padding=1)
        self.c_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.c_3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.c_4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.c_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.c_5 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.c_6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.c_7 = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=1)

        self.c_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.c_8 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.c_9 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.c_10 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1, padding=1)

        self.c_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.c_11 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.c_12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.c_13 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1, padding=1)
        self.c_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

        self.conv = torch.nn.Sequential(
            self.c_1,
            self.c_2,
            self.c_mx_1,
            self.c_3,
            self.c_4,
            self.c_mx_2,
            self.c_5,
            self.c_6,
            self.c_7,
            self.c_mx_3,
            self.c_8,
            self.c_9,
            self.c_10,
            self.c_mx_4,
            self.c_11,
            self.c_12,
            self.c_13,
            self.c_mx_5,
        )

        self.fc_net = torch.nn.Sequential(
            self.fc1, self.re1, self.fc2, self.re2, self.fc3, self.re3
        )

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""
        pass


class VGGD(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG D Model."""
        super(VGGD, self).__init__()

        self.d_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.d_2 = torch.nn.Conv2d(64, 64, (3, 3), strid=1, padding=1)
        self.d_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.d_3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.d_4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.d_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.d_5 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.d_6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.d_7 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)

        self.d_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.d_8 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.d_9 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.d_10 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        self.d_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.d_11 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.d_12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.d_13 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.d_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

        self.conv = torch.nn.Sequential(
            self.d_1,
            self.d_2,
            self.d_mx_1,
            self.d_3,
            self.d_4,
            self.d_mx_2,
            self.d_5,
            self.d_6,
            self.d_7,
            self.d_mx_3,
            self.d_8,
            self.d_9,
            self.d_10,
            self.d_mx_4,
            self.d_11,
            self.d_12,
            self.d_13,
            self.d_mx_5,
        )
        self.fc_net = torch.nn.Sequential(
            self.fc1, self.re1, self.fc2, self.re2, self.fc3, self.re3
        )

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""
        pass


class VGGE(torch.nn.Module):
    """VGG Implementation E Class."""

    def __init__(self):
        """Initialize VGG Model."""
        super(VGGE, self).__init__()
        self.e_1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.e_2 = torch.nn.Conv2d(64, 64, (3, 3), strid=1, padding=1)
        self.e_mx_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.e_3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.e_4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.e_mx_2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.e_5 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.e_6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.e_7 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.e_8 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)

        self.e_mx_3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.e_9 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_10 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_11 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        self.e_mx_4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.e_13 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_14 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_15 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.e_16 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)

        self.e_mx_5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = torch.nn.Linear(4096, 4096)
        self.re1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.re3 = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

        self.conv = torch.nn.Sequential(
            self.e_1,
            self.e_2,
            self.e_mx_1,
            self.e_3,
            self.e_4,
            self.e_mx_2,
            self.e_5,
            self.e_6,
            self.e_7,
            self.e_8,
            self.e_mx_3,
            self.e_9,
            self.e_10,
            self.e_11,
            self.e_12,
            self.e_mx_4,
            self.e_13,
            self.e_14,
            self.e_15,
            self.e_16,
            self.e_mx_5,
        )
        self.fc_net = torch.nn.Sequential(
            self.fc1, self.re1, self.fc2, self.re2, self.fc3, self.re3
        )

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""
        pass
