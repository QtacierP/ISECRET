# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding, norm_layer, use_bias, use_dropout=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding (nn.Padding)  -- the instance of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [padding(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [padding(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



