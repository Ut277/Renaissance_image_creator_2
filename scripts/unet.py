import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            CBR(in_channels, 64),
            CBR(64, 128),
            nn.MaxPool2d(2)
        )
        self.middle = CBR(128, 128)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            CBR(128, 64),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x