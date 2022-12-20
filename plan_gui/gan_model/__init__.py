import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
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

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), 
            nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="leaky", use_dropout=False)
        self.up2 = Block(
            features * 8 * 2, features * 4, down=False, act="leaky", use_dropout=False
        )
        self.up3 = Block(
            features * 4 * 2, features * 2, down=False, act="leaky", use_dropout=False
        )
        self.up4 = Block(
            features * 2 * 2, features, down=False, act="leaky", use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1)
        )

        self.lstm = self._rblock(image_size ** 2, image_size ** 2, 1)

        self.linear = self._lblock(in_channels * image_size ** 2, in_channels * image_size ** 2)

        self.sigmoid = nn.Sigmoid()

    def _lblock(self, in_channels, out_channels):
      return nn.Sequential(
          nn.Linear(in_channels, out_channels),
          nn.BatchNorm1d(out_channels),
      )
        
    
    def _rblock(self, in_channels, out_channels, layers):
      return nn.LSTM(in_channels, out_channels, layers, batch_first=True)

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        bottleneck = self.bottleneck(d4)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d4], 1))
        up3 = self.up3(torch.cat([up2, d3], 1))
        up4 = self.up4(torch.cat([up3, d2], 1))

        x = self.final_up(torch.cat([up4, d1], 1))

        batch_size = x.size(0)
        
        h_0 = torch.zeros(1, batch_size, self.image_size ** 2).to('cpu')
        c_0 = torch.zeros(1, batch_size, self.image_size ** 2).to('cpu')

        x = x.reshape(batch_size, self.in_channels, self.image_size ** 2)

        x, (hn, cn) = self.lstm(x, (h_0, c_0))

        x = x.reshape(batch_size, self.in_channels * self.image_size ** 2)

        x = self.linear(x)

        x = self.sigmoid(x)

        x = x.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        return x