import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import utils as vutils
from model import common
from model.utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding

# Pyramid Attention
class PyramidAttention(nn.Module):
    def __init__(self, scale=[1,0.9,0.8,0.7,0.6], res_scale=1, channel=64, reduction=2, ksize=3, \
                stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = common.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = common.BasicBlock(conv,channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv,channel, channel,1,bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        #theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base,1,dim=0)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bilinear')
            #feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            #sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            #sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            #group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, C, k, k]
            #normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)

            yi = F.softmax(yi*self.softmax_scale, dim=1)
            
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)/4.
            y.append(yi)
      
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        
        return y

# Patch-based Correlation Attention
class PatchCorrAttention(nn.Module):
    def __init__(self, channel=64, reduction=2, kernel_size=3, stride=1, softmax_scale=10, res_scale=1, conv=common.default_conv):
        super(PatchCorrAttention, self).__init__()

        self.softmax_scale = softmax_scale
        self.stride = stride
        self.res_scale = res_scale
        self.kernel_size = kernel_size

        self.unfold1 = nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.unfold2 = nn.Unfold(kernel_size=kernel_size, stride=stride)

        self.conv_match1 = common.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv,channel,channel//reduction, 1, bn=False, act=nn.PReLU())

        self.conv_assembly = common.BasicBlock(conv, channel//reduction, channel, 1, bn=False, act=nn.PReLU())

        self.channel_reduced = channel//reduction


    def forward(self, input):

        ref = input

        batches = input.size()[0]

        base = self.conv_match1(ref)

        inter_channel = base.size()[1]

        base_max = torch.max(torch.sqrt(reduce_sum(torch.pow(base, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)))

        base_normed = base/base_max

        base_normed = self.unfold1(base_normed)
        base = self.unfold1(base)

        ref_down = F.interpolate(input, scale_factor=0.5, mode='bilinear')

        w = self.conv_match2(ref_down)

        w_max = torch.max(torch.sqrt(reduce_sum(torch.pow(w, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)))
       
        w_normed = w/w_max

        w_normed = self.unfold2(w_normed)
        w = self.unfold2(w)

        base_patches = base.size()[-1]
        w_patches = w.size()[-1]

        patch_all = torch.cat([base, w], dim=-1).view(batches, inter_channel, self.kernel_size**2, -1)
        
        patch_all_normed = torch.cat([base_normed, w_normed], dim=-1).view(batches, inter_channel, self.kernel_size**2, -1)
        #correlation
        yi = torch.matmul(patch_all_normed.permute(0, 1, -1, -2), patch_all_normed)

        #softmax on correlation matrix
        yi = F.softmax(yi*self.softmax_scale, dim=-2) #instead of -1
                
        out = torch.matmul(patch_all, yi)#/self.channel_reduced
        out = out[:, :, :, :base_patches].view(batches, -1, base_patches)

        fold = nn.Fold(output_size=ref.size()[-2:], kernel_size=self.kernel_size, stride=self.stride)
        
        out = fold(out)/float(self.kernel_size**2)

        y = self.conv_assembly(out)

        y = y + self.res_scale*input
        
        return y
