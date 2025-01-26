"""VGG Classes and methods."""

import torch


class VGGA(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG A Model."""
        super(VGGA, self).__init__()

        self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
        self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""


        pass

class VGGALrn(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG A-lrn Model."""
        super(VGGALrn, self).__init__()

        self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
        self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""


        pass

class VGGB(torch.nn.Module):
    """VGG Implementation Class."""

    def __init__(self):
        """Initialize VGG B Model."""
        super(VGGB, self).__init__()

        self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
        self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

    def forward(self, x: torch.Tensor):
        """Model Forward Pass."""


        pass


class VGGC(torch.nn.Module):
    """VGG Implementation Class."""

def __init__(self):
    """Initialize VGG C Model."""
    super(VGGC, self).__init__()

    self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
    self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
    self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

def forward(self, x: torch.Tensor):
    """Model Forward Pass."""


    pass

class VGGD(torch.nn.Module):
    """VGG Implementation Class."""

def __init__(self):
    """Initialize VGG D Model."""
    super(VGGD, self).__init__()

    self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
    self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
    self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

def forward(self, x: torch.Tensor):
    """Model Forward Pass."""


    pass


class VGGE(torch.nn.Module):
    """VGG Implementation E Class."""

def __init__(self):
    """Initialize VGG Model."""
    super(VGGE, self).__init__()

    self.a_1 = torch.nn.Conv2d(3, 64, (3,3), stride=1, padding=1)
    self.a_mx_1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
    self.a_2 = torch.nn.Conv2d(64, 128, (3,3), strid=1, padding=1)

def forward(self, x: torch.Tensor):
    """Model Forward Pass."""


    pass