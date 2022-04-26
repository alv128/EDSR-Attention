import torch
import torch.nn.functional as F
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride==1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


# Enhanced Spatial Attention
class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
  
    def forward(self, f):
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)#used for ESA_EDSR and ESA_EDSR_GFS
        #v_max = F.max_pool2d(c1, kernel_size=3, stride=1)
        v_range = self.relu(self.conv_max(v_max))#3x3 kernel conv
        c3 = self.relu(self.conv3(v_range))#3x3 kernel conv
        c3 = self.conv3_(c3)#3x3 kernel conv
        c3 = F.interpolate(c3, (f.size(2), f.size(3)), mode='bilinear') 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return f * m


# Non-Local Enhanced Spatial Attention
class NLESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(NLESA, self).__init__()
        f = n_feats // 4
        self.inter_channel = f
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.theta = nn.Conv2d(f, f, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(f, f, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(f, f, kernel_size=1, stride=1, padding=0)

  
    def forward(self, f):
        batch_size, _, _, _ = f.size()
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=3, stride=1)
        v_range = self.relu(self.conv_max(v_max))#3x3 kernel conv
        c3 = self.relu(self.conv3(v_range))#3x3 kernel conv

        _, _, c3_2, c3_3 = c3.size()
        g_x = self.g(c3).view(batch_size, self.inter_channel, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(c3).view(batch_size, self.inter_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(c3).view(batch_size, self.inter_channel, -1)
        dot = torch.matmul(theta_x, phi_x)
        N = dot.size(-1)
        dot_N = dot / N
        y = torch.matmul(dot_N, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channel, c3_2, c3_3)


        # c3 = self.conv3_(c3)#3x3 kernel conv
        c3 = self.conv3_(y)
        c3 = F.interpolate(c3, (f.size(2), f.size(3)), mode='bilinear') 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return f * m


# Efficient Channal Attention Module
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# Spatial Attention via Efficient Channel Attention
class ESCA(nn.Module):

    def __init__(self, channel, k_size, reduction, conv=default_conv):
        super(ESCA, self).__init__()
        f = channel//reduction
        self.conv_red = nn.Conv2d(channel, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)

        self.conv_strided = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)

        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_spatial = nn.Sigmoid()

        self.conv4 = conv(f, channel, kernel_size=1)
    
    def forward(self, x):
        x_red = self.conv_red(x)

        #downsizing
        x_down = self.conv_strided(x_red)

        #further downsizing with pooling
        x_down = F.avg_pool2d(x_down, kernel_size=3, stride=1)

        x_max = self.relu(self.conv_max(x_down))
        x_= self.relu(self.conv3(x_max))

        x_group = torch.split(x_, 1, dim=0)
        x_restore = []
        for xi in x_group:
            N, C, H, W = xi.size()
            xi_ = xi.squeeze(0).view(C, 1, -1)
            xi_ = self.conv1(xi_)
            xi_ = self.sigmoid(xi_)
            xi_ = xi_.view(C, H, W).unsqueeze(0)
            x_restore.append(xi_*xi)
        
        x3 = torch.cat(x_restore, dim=0)
        x3 = self.conv3_(x3)
        x3 = F.interpolate(x3, (x.size(2), x.size(3)), mode='bilinear') 

        xf = self.conv_f(x_red)

        x4 = self.conv4(x3+xf)

        m = self.sigmoid_spatial(x4)

        return m*x