import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[0, :x.size(0), :x.size(1)]

class ControlledTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, control_size=32, nhead=12, num_layers=6, dropout=0.3):
        super(ControlledTransformerBlock, self).__init__()
        self.control_size = control_size
        self.out_channels = out_channels


        self.pos_encoder = PositionalEncoding(out_channels, max_len=1)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.control_encoder = nn.Sequential(
          nn.Linear(control_size, out_channels),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(True),
          nn.Linear(out_channels, out_channels),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(True),
          nn.Linear(out_channels, out_channels),
        )

        encoder_layers = TransformerEncoderLayer(out_channels, nhead=nhead, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


    def forward(self, x, x_c):
        x = self.conv(x)
        b, c, h, w = x.shape

        x = x.squeeze(-1).squeeze(-1)

        x = x + self.pos_encoder(x)
        x = x.reshape(b,c,h,w)

        x = x.view(b, c, -1).permute(2, 0, 1)
        x = x + self.control_encoder(x_c)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        encoder_layers = TransformerEncoderLayer(out_channels, nhead=nhead, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class ControlledBlock(nn.Module):
    def __init__(self, in_channels, out_channels, control_size=32, down=True, act="relu", use_dropout=False):
        super(ControlledBlock, self).__init__()

        self.out_channels = out_channels

        self.control_encoder = nn.Sequential(
          nn.Linear(control_size, out_channels*2),
          nn.BatchNorm1d(out_channels*2),
          nn.ReLU(True),
          nn.Linear(out_channels*2, out_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_join = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
        self.down = down

    def forward(self, x, x_c):
        x = self.conv(x)
        w_c = self.control_encoder(x_c)

        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)

        x = x + w_c
        x = x.permute(1, 2, 0).view(b, c, h, w)

        x = self.conv_join(x)

        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32, image_size=32, control_size=64):
        super().__init__()

        self.features = features
        self.in_channels = in_channels
        self.image_size = image_size

        self.initial_down = Block(
            in_channels, features, down=True, act="relu", use_dropout=False
        )
        self.down1 = Block(
            features, features * 2, down=True, act="relu", use_dropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="relu", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="relu", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="relu", use_dropout=False
        )

        self.final_down = Block (
            features * 8, features * 8, down=True, act="relu", use_dropout=False
        )

        self.transformer_1 = ControlledTransformerBlock(features * 8, features * 8, nhead=16, num_layers=32, control_size=control_size)

        self.up1 = ControlledBlock(
            features * 8, features * 8, down=False, act="relu", use_dropout=False, control_size=control_size
        )
        self.up2 = ControlledBlock(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False, control_size=control_size
        )
        self.up3 = ControlledBlock(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False, control_size=control_size
        )
        self.up4 = ControlledBlock(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False, control_size=control_size
        )
        self.final_up = ControlledBlock(
            features * 2, features, down=False, act="relu", use_dropout=False, control_size=control_size
        )

        self.linear = self._lblock(self.features * image_size ** 2, 2 * image_size ** 2)

    def _lblock(self, in_channels, out_channels):
         return nn.Sequential(
             nn.Linear(in_channels, out_channels),
         )

    def _rblock(self, in_channels, out_channels, layers):
      return nn.LSTM(in_channels, out_channels, layers, batch_first=True)

    def forward(self, x, x_control):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        final_down = self.final_down(d4)

        ct_1 = self.transformer_1(final_down, x_control)

        up1 = self.up1(ct_1, x_control)
        up2 = self.up2(torch.cat([up1, d4], dim=1), x_control)
        up3 = self.up3(torch.cat([up2, d3], dim=1), x_control)
        up4 = self.up4(torch.cat([up3, d2], dim=1), x_control)
        x = self.final_up(torch.cat([up4, d1], dim=1), x_control)

        batch_size = x.size(0)

        x = x.reshape(batch_size, self.features * self.image_size ** 2)

        x = self.linear(x)

        x = x.reshape(batch_size, 2, self.image_size, self.image_size)

        return x