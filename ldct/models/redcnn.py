"""RED-CNN: Residual Encoder-Decoder CNN (Chen et al., IEEE TMI 2017).

5 convolutions shrink the image (5x5 kernels, no padding), 5 transposed
convolutions grow it back. Three shortcuts add earlier features back in:
after conv2 and conv4 (to the matching decoder layers) and from the input
(to the output). With 96 channels it has 1,848,865 (about 1.85 M) parameters.

It is small and fast, so it is the reference model for testing the pipeline.
"""
import torch.nn as nn
import torch.nn.functional as F


class REDCNN(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        c = channels
        # encoder: each layer makes the image 4 pixels smaller (5x5, no padding)
        self.conv1 = nn.Conv2d(1, c, 5)
        self.conv2 = nn.Conv2d(c, c, 5)
        self.conv3 = nn.Conv2d(c, c, 5)
        self.conv4 = nn.Conv2d(c, c, 5)
        self.conv5 = nn.Conv2d(c, c, 5)
        # decoder: each layer makes the image 4 pixels bigger again
        self.deconv1 = nn.ConvTranspose2d(c, c, 5)
        self.deconv2 = nn.ConvTranspose2d(c, c, 5)
        self.deconv3 = nn.ConvTranspose2d(c, c, 5)
        self.deconv4 = nn.ConvTranspose2d(c, c, 5)
        self.deconv5 = nn.ConvTranspose2d(c, 1, 5)

    def forward(self, x):
        skip0 = x                          # shortcut from the input
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        skip2 = out                        # shortcut after conv2
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        skip4 = out                        # shortcut after conv4
        out = F.relu(self.conv5(out))

        out = self.deconv1(out) + skip4
        out = self.deconv2(F.relu(out))
        out = self.deconv3(F.relu(out)) + skip2
        out = self.deconv4(F.relu(out))
        out = self.deconv5(F.relu(out)) + skip0
        return F.relu(out)


def build(channels=96):
    return REDCNN(channels=channels)
