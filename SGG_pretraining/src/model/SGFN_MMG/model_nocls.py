import random

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from clip_adapter.model import AdapterModel
from src.model.losses.dec_loss import DECLoss
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMG import MMG
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti)

from src.utils import op_utils
from src.model.BT_with_CRR import BarlowTwinsWithCodingRate, BarlowTwins_CR_GeomAwareClustering

from sklearn.cluster import KMeans

class Mmgnet(BaseModel):
    def __init__(self, config, dim_descriptor=11):
        '''
        3d cat location, 2d
        '''

        super().__init__('Mmgnet', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3

        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.flow = 'target_to_source'
        dim_point_feature = 768
        self.momentum = 0.1
        self.model_pre = None

        self.obj_lambda_ = 0.0031
        self.edge_lambda_ = 0.0031
        self.triplet_lambda_ = 0.0011

        self.bt_w_crr_loss_obj = BarlowTwinsWithCodingRate(lambd=self.obj_lambda_)
        self.bt_w_crr_loss_edge = BarlowTwinsWithCodingRate(lambd=self.edge_lambda_)
        self.bt_w_crr_loss_triplet = BarlowTwinsWithCodingRate(lambd=self.triplet_lambda_)

        self.dec_loss = DECLoss(20,16)

        # self.loss_triplet = BarlowTwins_CR_GeomAwareClustering(lambd=self.triplet_lambda_)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        self.clip_model.requires_grad_(False)

        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)

        total_params1 = sum(p.numel() for p in self.obj_encoder.parameters() if p.requires_grad)

        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)

        total_params2 = sum(p.numel() for p in self.rel_encoder_3d.parameters() if p.requires_grad)

        self.mmg = MMG(
            dim_node=512,
            dim_edge=512,
            dim_atten=256, #self.mconfig.DIM_ATTEN
            depth=2, #self.mconfig.N_LAYERS
            num_heads=8, #self.mconfig.NUM_HEADS
            aggr="max", #self.mconfig.GCN_AGGR
            flow=self.flow,
            attention="fat",#self.mconfig.ATTENTION
            use_edge=True,#self.mconfig.USE_GCN_EDGE
            DROP_OUT_ATTEN=0.5)#self.mconfig.DROP_OUT_ATTEN

        total_params3 = sum(p.numel() for p in self.mmg.parameters() if p.requires_grad)

        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512 + 256, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.mlp_obj_forBT = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024)
        )

        self.mlp_edge_forBT = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024)
        )

        self.triplet_projector_3d_forclip = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )

        self.triplet_projector_3d_forBT = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024)
        )

        # self.triplet_projector_3d_forBT = torch.nn.Sequential(
        #     torch.nn.Linear(16, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 1024),
        #     torch.nn.BatchNorm1d(1024)
        # )

        self.all_triplet_low_feats = []

        total_params4 = sum(p.numel() for p in self.mlp_3d.parameters() if p.requires_grad)

        total_param = total_params1+total_params2+total_params3+total_params4

        print("total params of the SSG model: "+str(total_param))

        self.init_weight()

        mmg_obj, mmg_rel = [], []
        for name, para in self.mmg.named_parameters():
            if 'nn_edge' in name:
                mmg_rel.append(para)
            else:
                mmg_obj.append(para)

        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_3d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params': self.triplet_projector_3d_forDEC.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},
            {'params': self.triplet_projector_3d_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params': self.mlp_obj_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params': self.mlp_edge_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params':mmg_obj, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

        self.act = False

        one_epoch_iters = 32

        for i in range(one_epoch_iters*20):
            self.lr_scheduler.step()

    def init_weight(self,):
        torch.nn.init.xavier_uniform_(self.mlp_3d[0].weight)

    def update_model_pre(self, new_model):
        self.model_pre = new_model
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats

    
    def forward(self, obj_points, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        
        obj_feature = self.mlp_3d(obj_feature)

        if True :#self.mconfig.USE_SPATIAL
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=-1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)

        rel_feature_3d = self.rel_encoder_3d(edge_feature)

        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg(obj_feature, rel_feature_3d, edge_indices, batch_ids, obj_center, descriptor.clone(), istrain=istrain)

        triplet_feats = self.generate_object_pair_features(gcn_obj_feature_3d, gcn_edge_feature_3d,
                                                               edge_indices)

        triplet_feats_clip = self.triplet_projector_3d_forclip(triplet_feats)

        # triplet_feats_DEC = self.triplet_projector_3d_forDEC(triplet_feats)

        triplet_feats_BT = triplet_feats

        gcn_obj_feature_3d_BT = self.mlp_obj_forBT(gcn_obj_feature_3d)
        gcn_edge_feature_3d_BT = self.mlp_edge_forBT(gcn_edge_feature_3d)

        if istrain:
            return gcn_obj_feature_3d_BT, gcn_edge_feature_3d_BT, triplet_feats_clip, triplet_feats_BT, edge_feature
        else:
            return gcn_obj_feature_3d_BT, gcn_edge_feature_3d_BT, triplet_feats_clip, triplet_feats_BT, gcn_obj_feature_3d, gcn_edge_feature_3d

    def generate_clip_plabel(self,clip_feature_labels):

        clip_features_plabel = []

        for clip_label_list in clip_feature_labels:
            for clip_label in clip_label_list:
                clip_features_plabel.append(random.choice(clip_label))

        text_inputs = clip.tokenize(clip_features_plabel).to(self.device)
        clip_text_features = self.clip_model.encode_text(text_inputs)

        return clip_text_features


    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def process_train(self, obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, iters=None, clip_feature_labels=None, batch_ids=None):
        self.iteration +=1

        """
        all key feature shapes:
            clip_plabels: 311,512
            
            gcn_obj_feature_3d: 453,512
            
            gcn_edge_feature_3d: 311,512
        """
        if clip_feature_labels != None:
            with torch.no_grad():
                clip_plabels = self.generate_clip_plabel(clip_feature_labels)


        gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, edge_geom = self(obj_points, edge_indices.t().contiguous(), descriptor=descriptor, batch_ids=batch_ids, istrain=True)

        gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, rot_edge_geom = self(rot_obj_points, edge_indices.t().contiguous(), descriptor=rot_descriptor, batch_ids=batch_ids,
                                                       istrain=True)

        # loss_cls, _ = self.dec_loss(triplet_feats_DEC)


        """
        part I: compute text alignment loss
        """
        # if clip_feature_labels != None:
        #
        #     ori_triplet_feats_clip = ori_triplet_feats_clip / ori_triplet_feats_clip.norm(dim=-1, keepdim=True)
        #     rot_triplet_feats_clip = rot_triplet_feats_clip / rot_triplet_feats_clip.norm(dim=-1, keepdim=True)
        #
        #     ori_text_align_loss = F.l1_loss(ori_triplet_feats_clip, clip_plabels)
        #     rot_text_align_loss = F.l1_loss(rot_triplet_feats_clip, clip_plabels)

        """
        part II: compute diag loss
        """

        N,C,_ = edge_geom.shape

        #z_a, z_b, geom_a, geom_b, global_step
        self_loss_obj,_,_ = self.bt_w_crr_loss_obj(gcn_obj_feature_3d, gcn_rot_obj_feature_3d)
        self_loss_edge,_,_ = self.bt_w_crr_loss_edge(gcn_edge_feature_3d, gcn_rot_edge_feature_3d)
        self_loss_triplet,_,_ = self.bt_w_crr_loss_triplet(ori_triplet_feats_BT, rot_triplet_feats_BT)

        if clip_feature_labels != None:
            total_loss = 12 * ori_text_align_loss + 12 * rot_text_align_loss + self_loss_edge * 0.013 + self_loss_obj * 0.013 + self_loss_triplet*0.013
        else:
            if epoch>15:
                loss_cls,_ = self.dec_loss(triplet_feats_DEC)
                loss_rot_cls,_ = self.dec_loss(rot_triplet_feats_DEC)
                total_loss = self_loss_edge * 0.1 + self_loss_obj * 0.1 + self_loss_triplet*0.08+10*loss_cls+10*loss_rot_cls
                print("loss_cls: "+str((10*loss_cls+10*loss_rot_cls).item()))
            else:
                total_loss = self_loss_edge * 0.1 + self_loss_obj * 0.1 + self_loss_triplet * 0.08

        self.backward(total_loss)

        print("total loss:"+str(total_loss.item()))

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

    def process_valid(self, obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices,
                      clip_feature_labels=None, batch_ids=None):
        self.iteration += 1

        """
        all key feature shapes:
            clip_plabels: 311,512

            gcn_obj_feature_3d: 453,512

            gcn_edge_feature_3d: 311,512
        """
        if clip_feature_labels != None:
            with torch.no_grad():
                clip_plabels = self.generate_clip_plabel(clip_feature_labels)

        gcn_obj_feature_3d, gcn_edge_feature_3d, ori_triplet_feats_clip, ori_triplet_feats_BT, gcn_obj_feat, gcn_edge_feat = self(obj_points,
                                                                                                     edge_indices.t().contiguous(),
                                                                                                     descriptor,
                                                                                                     batch_ids,
                                                                                                    )

        gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_clip, rot_triplet_feats_BT, gcn_obj_feat_rot, gcn_edge_feat_rot = self(
            rot_obj_points, edge_indices.t().contiguous(), rot_descriptor, batch_ids,
            )

        # return gcn_obj_feat, gcn_edge_feat, gcn_obj_feat_rot, gcn_edge_feat_rot

        return gcn_obj_feature_3d, gcn_edge_feature_3d, gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d
