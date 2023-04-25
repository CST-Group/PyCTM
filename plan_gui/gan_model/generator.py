import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pe = nn.Parameter(torch.zeros(1, 1, out_channels))
        encoder_layers = TransformerEncoderLayer(out_channels, nhead=nhead, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)
        x = x + self.pe
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
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, image_size=32):
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

        self.bottleneck_core = nn.Sequential(
            Block(features * 8, features * 8, down=True, act="relu", use_dropout=False),
            TransformerBlock(features * 8, features*8, nhead=8, num_layers=8)
        )

        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up2 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up3 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up4 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )
        self.final_up = Block(
            features * 2, features, down=False, act="relu", use_dropout=False
        )
        
        self.linear = self._lblock(features * image_size ** 2, 2 * image_size ** 2)

        # self.sigmoid = nn.Sigmoid()

    def _lblock(self, in_channels, out_channels):
         return nn.Sequential(
             nn.Linear(in_channels, out_channels),
         )


    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        bottleneck_core = self.bottleneck_core(d4)
        up1 = self.up1(bottleneck_core)
        up2 = self.up2(torch.cat([up1, d4], 1))
        up3 = self.up3(torch.cat([up2, d3], 1))
        up4 = self.up4(torch.cat([up3, d2], 1))

        x = self.final_up(torch.cat([up4, d1], 1))

        batch_size = x.size(0)

        x = x.reshape(batch_size, self.features * self.image_size ** 2)

        x = self.linear(x)

        # x = self.sigmoid(x)

        x = x.reshape(batch_size, 2, self.image_size, self.image_size)

        return x