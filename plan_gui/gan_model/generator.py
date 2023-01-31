import torch
import torch.nn as nn
from torch.nn import Parameter

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()
        
        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key) # batch matrix-matrix product
        
        attention = self.softmax(energy) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        
        # Add attention weights onto input
        out = self.gamma*out + x
        return out
        # return out, attention


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"))
            if down
            else SpectralNorm(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),

            SelfAttention(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        # self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        # return self.dropout(x) if self.use_dropout else x
        return x

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
        
        h_0 = torch.zeros(1, batch_size, self.image_size ** 2)
        c_0 = torch.zeros(1, batch_size, self.image_size ** 2)

        x = x.reshape(batch_size, self.in_channels, self.image_size ** 2)

        x, (hn, cn) = self.lstm(x, (h_0, c_0))

        x = x.reshape(batch_size, self.in_channels * self.image_size ** 2)

        x = self.linear(x)

        x = self.sigmoid(x)

        x = x.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        return x