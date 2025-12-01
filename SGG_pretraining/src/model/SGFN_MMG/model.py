import random

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from clip_adapter.model import AdapterModel
from src.model.losses.dec_loss import DECLoss
from src.model.losses.dec_model import DEC
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMG import MMG
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti)
from src.model.edge_Encoder import TransformerEncoder4Layers

from src.utils import op_utils
from src.model.BT_with_CRR import BarlowTwinsWithCodingRate, BarlowTwins_CR_GeomAwareClustering
from src.model.pointTransformer.new_pointTrans import PointTransformerEncoder
from src.model.depthContrast.models.trunks.pointnet import PointNet,clean_weights
from sklearn.cluster import KMeans
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR
import math
def get_warmup_cosine_scheduler(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=0.5,
    min_lr=0.0,
    last_epoch=-1
):
    """
    创建一个学习率调度器，先线性预热，然后进行余弦退火
    
    参数:
        optimizer: 优化器
        num_warmup_steps: 预热阶段的步数
        num_training_steps: 总训练步数
        num_cycles: 余弦函数的波数 (0.5表示半个余弦波)
        min_lr: 最低学习率 (相对于初始lr的比例)
        last_epoch: 上一轮的索引(用于继续训练)
    """
    def lr_lambda(current_step):
        # 预热阶段: 线性增长到基础学习率
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 预热后: 余弦退火
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # 将学习率从1降到min_lr
        return min_lr + (1.0 - min_lr) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

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

        initial_cluster_centers = torch.zeros(
            20,
            64,
            dtype=torch.float
        ).cuda()
        torch.nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = Parameter(initial_cluster_centers)

        # self.loss_triplet = BarlowTwins_CR_GeomAwareClustering(lambd=self.triplet_lambda_)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        self.clip_model.requires_grad_(False)

        # Object Encoder
        # self.obj_encoder = PointNetfeat(
        #     global_feat=True,
        #     batch_norm=with_bn,
        #     point_size=dim_point,
        #     input_transform=False,
        #     feature_transform=mconfig.feature_transform,
        #     out_size=dim_point_feature)

        # self.obj_encoder = PointNet(scale=1)
        self.obj_encoder = PointTransformerEncoder(latent_dim=768)
        
        total_params1 = sum(p.numel() for p in self.obj_encoder.parameters() if p.requires_grad)

        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=11,
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
            torch.nn.Linear(768, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.mlp_obj_forBT = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048)
        )

        self.mlp_edge_forBT = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048)
        )

        self.mlp_triplet_forBT = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048)
        )

        self.triplet_projector_3d_forDEC = torch.nn.Sequential(
            torch.nn.Linear(512*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
        )

        self.triplet_projector_3d_forCLIP = torch.nn.Sequential(
            torch.nn.Linear(512*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
        )
        
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
            {'params': self.triplet_projector_3d_forDEC.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},
             {'params': self.triplet_projector_3d_forCLIP.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},
            {'params': self.cluster_centers, 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params': self.mlp_obj_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params': self.mlp_edge_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params': self.mlp_triplet_forBT.parameters(), 'lr': float(config.LR),
             'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD},

            {'params':mmg_obj, 'lr':float(config.LR)/2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel, 'lr':float(config.LR)/4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        
        one_epoch_iters = 47
        self.lr_scheduler = get_warmup_cosine_scheduler(self.optimizer,one_epoch_iters*30,self.config.max_iteration) #CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1) #
        self.optimizer.zero_grad()

        self.act = False

        for i in range(one_epoch_iters*0):
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

        triplet_feats_clip = self.triplet_projector_3d_forCLIP(triplet_feats)

        triplet_feats_DEC = self.triplet_projector_3d_forDEC(triplet_feats)

        triplet_feats_high = self.mlp_triplet_forBT(triplet_feats)

        gcn_obj_feature_3d_BT = self.mlp_obj_forBT(gcn_obj_feature_3d)
        gcn_edge_feature_3d_BT = self.mlp_edge_forBT(gcn_edge_feature_3d)

        if istrain:
            return gcn_obj_feature_3d_BT, gcn_edge_feature_3d_BT, triplet_feats_DEC, triplet_feats_high, edge_feature, triplet_feats_clip
        else:
            return gcn_obj_feature_3d_BT, gcn_edge_feature_3d_BT, triplet_feats_DEC, triplet_feats_high, gcn_obj_feature_3d, gcn_edge_feature_3d

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

    def process_cluster(self, obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, isMulti=False, clip_feature_labels=None, batch_ids=None, epoch=None):
        self.iteration +=1

        gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, edge_geom = self(obj_points, edge_indices.t().contiguous(), descriptor=descriptor, batch_ids=batch_ids, istrain=True)
        if isMulti:
            gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, rot_edge_geom = self(
                rot_obj_points, edge_indices.t().contiguous(), descriptor=rot_descriptor, batch_ids=batch_ids,
                istrain=True)
            return triplet_feats_DEC,rot_triplet_feats_DEC


        return triplet_feats_DEC

    def model_output(self, obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, iters=None, clip_feature_labels=None, batch_ids=None, epoch=None):
        self.iteration +=1


        gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, edge_geom, triplet_feats_clip = self(obj_points, edge_indices.t().contiguous(), descriptor=descriptor, batch_ids=batch_ids, istrain=True)

        gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, rot_edge_geom, rot_triplet_feats_clip = self(rot_obj_points, edge_indices.t().contiguous(), descriptor=rot_descriptor, batch_ids=batch_ids,
                                                       istrain=True)

        if clip_feature_labels != None:
            with torch.no_grad():
                clip_plabels = self.generate_clip_plabel(clip_feature_labels)
                return gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT,triplet_feats_clip, rot_triplet_feats_clip,clip_plabels
        
        return gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, triplet_feats_clip, rot_triplet_feats_clip,clip_plabels

    def process_train(self, obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, iters=None, clip_feature_labels=None, batch_ids=None, epoch=None):
        self.iteration +=1

      
        if clip_feature_labels != None:
            with torch.no_grad():
                clip_plabels = self.generate_clip_plabel(clip_feature_labels)


        gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, edge_geom = self(obj_points, edge_indices.t().contiguous(), descriptor=descriptor, batch_ids=batch_ids, istrain=True)

        gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, rot_edge_geom = self(rot_obj_points, edge_indices.t().contiguous(), descriptor=rot_descriptor, batch_ids=batch_ids,
                                                       istrain=True)

        # loss_cls, _ = self.dec_loss(triplet_feats_DEC)

        # if epoch > 20 and epoch % 5 == 0 and self.act == True:
        #     with torch.no_grad():
        #         self.all_triplet_low_feats.append(triplet_feats_DEC.detach().cpu())
        #     self.act = False
        #
        # if epoch > 20  and  epoch % 5 != 0 and self.act == False :
        #     if not self.initial_param:
        #         cluster_feats = torch.cat(self.all_triplet_low_feats)
        #         kmeans = KMeans(n_clusters=20, random_state=0).fit(cluster_feats)
        #         cluster_centers = kmeans.cluster_centers_
        #         cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
        #         self.dec_loss.cluster_centers = torch.nn.Parameter(cluster_centers)
        #         self.initial_param = True
        #         self.all_triplet_low_feats = []
        #     else:
        #         cluster_feats = torch.cat(self.all_triplet_low_feats)
        #         kmeans = KMeans(n_clusters=20, random_state=0).fit(cluster_feats)
        #         cluster_centers = kmeans.cluster_centers_
        #         cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
        #         self.dec_loss.cluster_centers = self.dec_loss.cluster_centers*0.9+cluster_centers*0.1
        #         self.all_triplet_low_feats = []
        #     self.act = True
        
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
        _,self_loss_obj,_ = self.bt_w_crr_loss_obj(gcn_obj_feature_3d, gcn_rot_obj_feature_3d)
        _,self_loss_edge,_ = self.bt_w_crr_loss_edge(gcn_edge_feature_3d, gcn_rot_edge_feature_3d)
        _,self_loss_triplet,_ = self.bt_w_crr_loss_triplet(ori_triplet_feats_BT, rot_triplet_feats_BT)

        if clip_feature_labels != None:
            total_loss = 12 * ori_text_align_loss + 12 * rot_text_align_loss + self_loss_edge * 0.013 + self_loss_obj * 0.013 + self_loss_triplet*0.013
        else:
            if epoch>50 and self.act==True:
                loss_cls,_ = self.dec_loss(triplet_feats_DEC)
                loss_rot_cls,_ = self.dec_loss(rot_triplet_feats_DEC)
                total_loss = self_loss_edge * 0.1 + self_loss_obj * 0.1 + self_loss_triplet*0.08 + 10*loss_cls + 10*loss_rot_cls
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

        gcn_obj_feature_3d, gcn_edge_feature_3d, ori_triplet_feats_DEC, ori_triplet_feats_BT, gcn_obj_feat, gcn_edge_feat = self(obj_points,
                                                                                                     edge_indices.t().contiguous(),
                                                                                                     descriptor,
                                                                                                     batch_ids,
                                                                                                    )

        gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT, gcn_obj_feat_rot, gcn_edge_feat_rot = self(
            rot_obj_points, edge_indices.t().contiguous(), rot_descriptor, batch_ids,
            )

        # return gcn_obj_feat, gcn_edge_feat, gcn_obj_feat_rot, gcn_edge_feat_rot

        return gcn_obj_feature_3d, gcn_edge_feature_3d, gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, ori_triplet_feats_BT, rot_triplet_feats_BT
