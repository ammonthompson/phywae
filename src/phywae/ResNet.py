import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from phywae import utils
import random

class ResCnnEncoder(nn.Module):
    def __init__(self, data_channels, data_width, layer_params):
        print("using Resnet encoder")
        super().__init__()
        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        nl = len(out_channels)
        self.cnn_layers = nn.Sequential()
        self.conv_out_width = []

        in_channels = data_channels
        for i in range(nl):
            if i < (nl - 1):
                self.cnn_layers.add_module(
                    f"resblock_{i}",
                    ResidualBlockCNN(
                        in_channels = in_channels,
                        out_channels= out_channels[i],
                        kernel_size = kernel[i],
                        stride      = stride[i],
                        bias        = False,
                    ),
                )
            else: # if final layer, simple convolution layer (no norm or relu)
                self.cnn_layers.add_module(
                    f"conv1d_{i}",
                    nn.Conv1d(
                        in_channels = in_channels,
                        out_channels= out_channels[i],
                        kernel_size = kernel[i],
                        stride      = stride[i],
                        padding     = kernel[0]// 2,
                        bias        = True,
                    )
                )

            conv_out_shape = utils.get_outshape(self.cnn_layers, data_channels, data_width)
            self.conv_out_width.append(conv_out_shape[2])
            print(conv_out_shape)

            in_channels = out_channels[i]

    def forward(self, x):
        return self.cnn_layers(x)


class ResCnnDecoder(nn.Module):
    def __init__(self, 
                 encoder_layer_widths, 
                 latent_width, 
                 data_channels, 
                 data_width, 
                 layer_params,
                 num_chars,
                 char_type):
        
        super().__init__()

        print("using Resnet dencoder")


        out_channels = layer_params['out_channels']
        kernel       = layer_params['kernel']
        stride       = layer_params['stride']
        num_cnn_latent_channels = out_channels[-1]
        self.char_type          = char_type
        self.num_chars          = num_chars
        self.data_channels      = data_channels
        nl = len(out_channels)

        self.tcnn_layers = nn.Sequential()

        # first upsample block
        w_in = latent_width
        new_target_width = encoder_layer_widths[-2]
        pad, outpad = self._get_paddings(new_target_width, w_in, stride[-1], kernel[-1])
        self.tcnn_layers.add_module(
            "tresblock_0",
            ResidualBlockTransposeCNN(
                in_channels=out_channels[nl - 1],
                out_channels=out_channels[nl - 2],
                kernel_size=kernel[nl - 1],
                stride=stride[nl - 1],
                padding=pad,
                output_padding=outpad,
            ),
        )

        # middle blocks
        for i in range(nl - 2, 0, -1):
            outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
            w_in = outshape[2]
            new_target_width = encoder_layer_widths[i - 1]
            pad, outpad = self._get_paddings(new_target_width, w_in, stride[i], kernel[i])
            self.tcnn_layers.add_module(
                "tresblock_" + str(nl - i - 1),
                ResidualBlockTransposeCNN(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i - 1],
                    kernel_size=kernel[i],
                    stride=stride[i],
                    padding=pad,
                    output_padding=outpad,
                ),
            )

        # final decoder layer (no residual to mirror CnnDecoder headroom)
        outshape = utils.get_outshape(self.tcnn_layers, num_cnn_latent_channels, latent_width)
        width = outshape[2]
        new_target_width = data_width
        pad, outpad = self._get_paddings(new_target_width, width, stride[0], kernel[0])
        self.tcnn_layers.add_module(
            "struct_decoder_out",
            nn.ConvTranspose1d(
                in_channels=out_channels[0],
                out_channels=data_channels,
                kernel_size=kernel[0],
                stride=stride[0],
                padding=pad,
                output_padding=outpad,
            ),
        )

        self.char_head_out = _DecoderHead(num_chars, data_width)

    def forward(self, x):
        decoded_x = self.tcnn_layers(x)
        char_start_idx = decoded_x.shape[1] - self.num_chars

        decoded_phylo = decoded_x[:, :char_start_idx, :]

        if self.char_type == "categorical" and self.num_chars > 0 and self.data_channels > 2:
            decoded_char = self.char_head_out(decoded_x[:, char_start_idx:, :])
            final_decoded_x = torch.cat((decoded_phylo, decoded_char), dim=1)
        else:
            final_decoded_x = torch.cat((decoded_phylo, decoded_x[:, char_start_idx:, :]), dim=1)

        return final_decoded_x

    def _get_paddings(self, w_out_target, w_in, s, k, d=1):
        # convtranspose1d formula: 
        # w_out_target = (w_in - 1)s + d(k-1) + 1 + outpad - 2pad
        # returns the paddings necessary for target output width 
        E = s*(w_in - 1) + d*(k-1) + 1

        if (w_out_target - E) < 0:
            pad = (E - w_out_target)/2
            pad = int(pad + pad % 1)
            outpad = w_out_target - E + 2 * pad
        elif (w_out_target - E) >=0 and (w_out_target - E) <= (s - 1):
            outpad = w_out_target - E
            pad = 0
        else:
            pad = (E - w_out_target + s - 1)/2
            pad = int(pad + pad % 1)
            outpad = s-1

        return pad, outpad
       

class ResidualBlockCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),            
            # no change dims
            nn.Conv1d(out_channels, out_channels, 
                      kernel_size=kernel_size, stride=1, 
                      padding="same", bias=False),
            nn.BatchNorm1d(out_channels)
        )
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 
                          kernel_size=1, stride=stride, 
                          padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )


    def forward(self, x):
        out = self.cnn_layers(x)
        skip_out = self.skip(x)

        # When using even kernels or strides > 1, skip path and main path
        # can be off by one in width; pad/crop the skip path to match.
        if skip_out.shape[-1] != out.shape[-1]:
            diff = out.shape[-1] - skip_out.shape[-1]
            if diff > 0:
                skip_out = F.pad(skip_out, (0, diff))
            elif diff < 0:
                skip_out = skip_out[..., :out.shape[-1]]

        out = F.relu(out + skip_out)
        return out
    

class ResidualBlockTransposeCNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, output_padding):
        super().__init__()

        self.tcnn_layers = nn.Sequential(
            # Upsample
            nn.ConvTranspose1d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),  # keep it non-inplace for safety
            # no change dims
            nn.Conv1d(out_channels, out_channels,
                kernel_size=kernel_size, stride=1,
                padding="same", bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            # Match both channels and (when stride>1) length
            self.skip = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    padding=0, output_padding=output_padding,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.tcnn_layers(x)
        skip_out = self.skip(x)

        if skip_out.shape[-1] != out.shape[-1]:
            diff = out.shape[-1] - skip_out.shape[-1]
            if diff > 0:
                skip_out = F.pad(skip_out, (0, diff))
            elif diff < 0:
                skip_out = skip_out[..., :out.shape[-1]]

        out = F.relu(out + skip_out)
        return out


class _DecoderHead(nn.Module):
    def __init__(self, c, w):
        super().__init__()
        self.c = c
        self.w = w
        self.scale = nn.Parameter(torch.ones(c * w))

    def forward(self, x):
        flat_x = x.view((x.shape[0], self.c * self.w))
        scaled_flat_x = self.scale * flat_x
        scaled_x = scaled_flat_x.view(x.shape[0], self.c, self.w)
        return scaled_x
