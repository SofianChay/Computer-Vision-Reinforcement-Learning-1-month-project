import torch
from torch import nn
import torch.nn.functional as F



class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
                    nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels), 
                    nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    )
        return block


    def expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
                    nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels),
                    nn.ReLU(),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels),
                    nn.ReLU(mid_channels),
                    nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                    )
        return block

    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
                    nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels)
                    # the number of feature maps equal to the number of segments desired.
                    )
        return block

    def __init__(self, in_channel, out_channel):
        # out_channel represents number of segments desired
        super(UNet, self).__init__()
        # Encode (contraction)
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        # The number of feature maps after each block doubles so that architecture can learn the complex structures 

        # Bottleneck
        self.bottleneck = nn.Sequential(
                            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
                            nn.ReLU(),
                            nn.BatchNorm2d(512),
                            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
                            nn.ReLU(),
                            nn.BatchNorm2d(512),
                            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )


        # Decode (expansion)
        self.conv_decode3 = self.expansive_block(512, 256, 128) 
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer =  self.final_block(128, 64, out_channel)


    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        The function crop_and_concat appends the output 
        of contraction layer with the new expansion layer input.
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            if (bypass.size()[2] - upsampled.size()[2]) % 2 == 0:
                upsampled = F.pad(upsampled, (c, c, c, c))
            else:
                upsampled = F.pad(upsampled, (c+1, c, c+1, c))
        return torch.cat((upsampled, bypass), 1)


    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer
