import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import get_knn_pts, index_points
from einops import repeat

class WeightLayer(nn.Module):
    def __init__(self, args):
        super(WeightLayer, self).__init__()

        self.mlp_0 = Conv2D(3, args.feat_dim)
        self.mlp_1 = Conv2D(args.feat_dim, args.feat_dim)
        self.mlp_2 = Conv2D(args.feat_dim*2, 1, with_bn=False, with_relu=False)
        self.k = args.k
        self.act = torch.nn.Sigmoid()

    def forward(self, original_pts, query_pts, local_feat):
        B, _, N = query_pts.shape
        knn_pts, knn_idx = get_knn_pts(self.k, original_pts, query_pts, return_idx=True)
        patch_feat = index_points(local_feat, knn_idx)[:,:,:,:self.k//4] # b, c, n, k/2
        
        repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=self.k)
        relative_pts = knn_pts - repeat_query_pts # b 3 n k
        feat0 = self.mlp_0(relative_pts)# b c n k
        feat_g = torch.max(feat0, dim=3, keepdim=True)[0]# b c n 1
        relative_feat = self.mlp_1(feat0)[:,:,:,:self.k//4] # b c n k/4
        feat2 = torch.cat([relative_feat, feat_g.view(B, -1, N, 1).repeat(1, 1, 1, self.k//4),], dim=1) # b 2c n k/4
        weight = self.act(self.mlp_2(feat2)) # b 1 n k/4
        weight_d = 1-weight
        
        query_feat = weight_d * relative_feat + weight * patch_feat
        query_feat = torch.sum(query_feat, dim=-1) # (b, c, n)
        return query_feat

class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x
    
class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv2D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        """
            x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x