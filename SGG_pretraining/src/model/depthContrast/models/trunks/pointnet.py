# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
os.sys.path.append('/home/honsen/honsen/SceneGraph/baseline/src/model/depthContrast')
from models.trunks.mlp import MLP

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'pointnet2'))

from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class PointNet(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0, scale=1, use_mlp=False, mlp_dim=None):
        super().__init__()

        self.use_mlp = use_mlp
        self.sa1 = PointnetSAModuleVotes(
                npoint=64,
                radius=0.2,
                nsample=32,
                mlp=[input_feature_dim, 64*scale, 64*scale, 128*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=64,
                radius=0.8,
                nsample=16,
                mlp=[256*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=64,
                radius=1.2,
                nsample=16,
                mlp=[256*scale, 128*scale, 128*scale, 256*scale],
                use_xyz=True,
                normalize_xyz=True
            )

        if scale == 1:
            self.fp1 = PointnetFPModule(mlp=[256+256,512,512])
            self.fp2 = PointnetFPModule(mlp=[512+256,512,512])
        else:
            self.fp1 = PointnetFPModule(mlp=[256*scale+256*scale,256*scale,256*scale])
            self.fp2 = PointnetFPModule(mlp=[256*scale+256*scale,256*scale,256*scale])

        if use_mlp:
            self.head = MLP(mlp_dim)
            
        self.all_feat_names = [
            "sa1",
            "sa2",
            "sa3",
            "sa4",
            "fp1",
            "fp2",
        ]
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, out_feat_keys=["fp2"]):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        batch_size = pointcloud.shape[0]

        pointcloud = pointcloud.permute(0, 2, 1)  
        
        xyz, features = self._break_up_pc(pointcloud)

        features = None ### Do not use other info for now
        
        end_points = {}

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        end_points['fp1_features'] = features
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        out_feats = [None] * len(out_feat_keys)

        final_feat = []
        for key in out_feat_keys:
            feat = end_points[key+"_features"]
            org_feat = end_points[key+"_features"]
            nump = feat.shape[-1]
            feat = torch.squeeze(F.max_pool1d(feat, nump))
            ### Apply the head here
            if self.use_mlp:
                feat = self.head(feat)
            out_feats[out_feat_keys.index(key)] = feat
            
        return out_feats

def clean_weights(state_dict):
    """
    处理权重字典：
    1. 去除'module.trunk.0.'前缀
    2. 删除所有包含'module.trunk.1.'的权重
    
    参数:
        state_dict: 包含模型权重的字典
        
    返回:
        新字典，经过处理后
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # 删除包含'module.trunk.1.'的权重
        if 'module.trunk.1.' in key:
            continue
            
        # 去除'module.trunk.0.'前缀
        if key.startswith('module.trunk.0.'):
            new_key = key.replace('module.trunk.0.', '', 1)
        else:
            new_key = key
            
        new_state_dict[new_key] = value
    
    return new_state_dict


if __name__=='__main__':
    backbone_net = PointNet(input_feature_dim=0,scale=4).cuda()
    qwe=torch.load("/home/honsen/下载/checkpoint-ep150.pth.tar")
    
    stat_dict1 = clean_weights(qwe['model'])
    
    asd=backbone_net.load_state_dict(stat_dict1)
    
    
    
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,3).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
