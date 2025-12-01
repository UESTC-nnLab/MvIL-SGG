from tqdm import tqdm

from src.model.losses.dec_model import DEC
from src.dataset.patchdropOut import PointPatchDropout
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import copy
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.model.BT_with_CRR import BarlowTwins_loss, dec_loss, calculate_coding_rate_loss, BarlowTwinsWithCodingRate
from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg,collate_fn_mmg_single)
from src.dataset.dataset_builder import build_dataset,build_honsenDataset
from src.model.SGFN_MMG.model import Mmgnet
from src.utils import op_utils
from src.utils.eva_utils_acc import get_mean_recall,get_zero_shot_recall
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def remove_parameters_by_name(state_dict, keyword):
    """删除权重字典中名称包含keyword的参数"""
    keys_to_remove = [k for k in state_dict.keys() if keyword in k]
    for k in keys_to_remove:
        del state_dict[k]
    print(f"Removed keys: {keys_to_remove}")
    return state_dict

class MMGNet():
    def __init__(self, config):
        self.config = config
        self.model_name = 'HONSEN_sgg'
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        self.isInit=False
        
        self.dropout = PointPatchDropout(dropout_ratio=1, num_patches=2, patch_radius=0.4)
        self.obj_lambda_ = 0.0053
        self.edge_lambda_ = 0.0053
        self.triplet_lambda_ = 0.0053
        self.low_lambda_ = 0.0053
        self.bt_w_crr_loss_obj = BarlowTwinsWithCodingRate(lambd=self.obj_lambda_,isCR=False)
        self.bt_w_crr_loss_edge = BarlowTwinsWithCodingRate(lambd=self.edge_lambda_,isCR=False)
        self.bt_w_crr_loss_triplet = BarlowTwinsWithCodingRate(lambd=self.triplet_lambda_,isCR=False)
        self.bt_w_crr_loss_low = BarlowTwinsWithCodingRate(lambd=self.low_lambda_, isCR=False)
        self.p=None
        self.rot_p=None
        ''' Build dataset '''
        dataset = None
        if config.MODE  == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_honsenDataset()

        if config.MODE  == 'train':
            asd = len(self.dataset_train)
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)

        ''' Build Model '''
        self.model = Mmgnet(self.config).to(config.DEVICE)

        self.sample_counts = []

    def load(self, best=False):
        return self.model.load(best)
        
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids


    @torch.no_grad()
    def data_processing_train_honsen(self, items):
        obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, clip_feature_labels, batch_ids = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        rot_obj_points = rot_obj_points.permute(0, 2, 1).contiguous()
        obj_points, edge_indices, descriptor, rot_obj_points, rot_descriptor, batch_ids = \
            self.cuda(obj_points, edge_indices, descriptor, rot_obj_points, rot_descriptor, batch_ids)

        return obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, clip_feature_labels, batch_ids

    @torch.no_grad()
    def data_processing_train_honsen1(self, items):
        obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, batch_ids = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        rot_obj_points = rot_obj_points.permute(0, 2, 1).contiguous()
        obj_points, edge_indices, descriptor, rot_obj_points, rot_descriptor, batch_ids = \
            self.cuda(obj_points, edge_indices, descriptor, rot_obj_points, rot_descriptor, batch_ids)

        return obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, None, batch_ids

    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_ids = items
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_ids
          
    def train(self):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=128,
            num_workers=4, #self.config.WORKERS
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )
        
        # BATCH_SIZE = 16
        
        self.model.epoch = 0
        keep_training = True

        startEpoch= 300

        perEpoch = 10


        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
        useclip = False
        
        if False:
            obj_encoder_dict = torch.load("obj_encoder.pth")
            self.model.obj_encoder.load_state_dict(obj_encoder_dict)

            mmg_dict = torch.load("mmg.pth")
            self.model.mmg.load_state_dict(mmg_dict)

            rel_encoder_dict = torch.load("rel_encoder.pth")
            self.model.rel_encoder_3d.load_state_dict(rel_encoder_dict)

            mlp_dict = torch.load("mlp.pth")
            self.model.mlp_3d.load_state_dict(mlp_dict)

            mlp_edge_forBT_dict = torch.load(
                "mlp_edge_forBT.pth")
            self.model.mlp_edge_forBT.load_state_dict(mlp_edge_forBT_dict)

            mlp_obj_forBT_dict = torch.load("mlp_obj_forBT.pth")
            self.model.mlp_obj_forBT.load_state_dict(mlp_obj_forBT_dict)

            triplet_projector_3d_forBT_dict = torch.load("mlp_triplet_forBT.pth")
            self.model.mlp_triplet_forBT.load_state_dict(triplet_projector_3d_forBT_dict)



        for k, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"Para {k} need grad")

        ''' Train '''
        self.model.train()

        while(keep_training):

            if self.model.epoch > self.config.MAX_EPOCHES:#
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)
            
            if self.model.epoch >= startEpoch and self.model.epoch % perEpoch == 0 and self.isInit==False:
                # 阶段2: 初始化聚类中心，并开始联合训练
                print("\n--- Initializing Cluster Centers with K-means ---")
                self.model.eval()
                all_z64 = []
                with torch.no_grad():
                    for items in tqdm(loader):
                        obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, clip_feature_labels, batch_ids = self.data_processing_train_honsen1(items)
                        z64 = self.model.process_cluster(obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, batch_ids=batch_ids)
                        all_z64.append(z64.cpu())
                all_z64 = torch.cat(all_z64, dim=0).numpy()
                loader = iter(train_loader)
                kmeans = KMeans(n_clusters=20)
                kmeans.fit(all_z64)
                np.save("/home/honsen/honsen/SceneGraph/ourPretrained_SGG/src/cluster_center.npy", kmeans.cluster_centers_)
                # cluster_centers_ = np.load("/home/honsen/honsen/SceneGraph/ourPretrained_SGG/src/cluster_center.npy")
                # 将K-means找到的中心点赋值给模型中的聚类中心参数
                self.model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device='cuda')
                self.isInit=True
                
            
            if self.model.epoch>=startEpoch and self.model.epoch % perEpoch == 0 and self.isInit==True:
                    print("\n--- 计算当前所有数据的目标分布 p ---")
                 # 计算当前所有数据的目标分布 p
                    self.sample_counts = []
                    all_q = []
                    rot_all_q = []
                    self.model.eval()
                    with torch.no_grad():
                         for items in tqdm(loader):
                            obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, clip_feature_labels, batch_ids = self.data_processing_train_honsen1(items)
                            z64, rot_z64 = self.model.process_cluster(obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, isMulti=True, batch_ids=batch_ids)
                            # 计算软分配 q

                            dists = torch.sum((z64.unsqueeze(1) - self.model.cluster_centers) ** 2, dim=2)
                            q_batch = (1.0 + dists).pow_(-1)
                            q_batch = (q_batch.T / torch.sum(q_batch, dim=1)).T
                            all_q.append(q_batch)
                            
                            # 计算软分配 q_rot
                            rot_dists = torch.sum((rot_z64.unsqueeze(1) - self.model.cluster_centers) ** 2, dim=2)
                            rot_q_batch = (1.0 + rot_dists).pow_(-1)
                            rot_q_batch = (rot_q_batch.T / torch.sum(rot_q_batch, dim=1)).T
                            rot_all_q.append(rot_q_batch)

                    for q_i in all_q:
                        self.sample_counts.append(q_i.shape[0])

                    rot_all_q = torch.cat(rot_all_q, dim=0).cuda()
                    all_q = torch.cat(all_q, dim=0).cuda()
                    # 计算目标分布 p
                    p = all_q**2 / all_q.sum(0)
                    self.p = (p.T / p.sum(1)).T

                    # 计算目标分布 rot_p
                    rot_p = rot_all_q**2 / rot_all_q.sum(0)
                    self.rot_p = (rot_p.T / rot_p.sum(1)).T
                    self.model.train()
                    loader = iter(train_loader)
                    
            for batch_idx, items in enumerate(loader):
                ''' get data '''
                print("------training------")
                obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices, clip_feature_labels, batch_ids = self.data_processing_train_honsen1(items)
                rot_obj_points = self.dropout(rot_obj_points.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
                
                gcn_obj_feature_3d, gcn_edge_feature_3d, triplet_feats_DEC, ori_triplet_feats_BT, gcn_rot_obj_feature_3d, gcn_rot_edge_feature_3d, rot_triplet_feats_DEC, rot_triplet_feats_BT\
                ,triplet_feats_clip, rot_triplet_feats_clip,clip_plabels=self.model.model_output(obj_points, descriptor, rot_obj_points, rot_descriptor, edge_indices,
                                       clip_feature_labels=clip_feature_labels, batch_ids=batch_ids)

                self_loss_obj = self.bt_w_crr_loss_obj(gcn_obj_feature_3d, gcn_rot_obj_feature_3d)
                self_loss_edge = self.bt_w_crr_loss_edge(gcn_edge_feature_3d, gcn_rot_edge_feature_3d)
                self_loss_triplet = self.bt_w_crr_loss_triplet(ori_triplet_feats_BT, rot_triplet_feats_BT)            
                
                triplet_feats_clip = triplet_feats_clip / triplet_feats_clip.norm(dim=-1, keepdim=True)
                rot_triplet_feats_clip = rot_triplet_feats_clip / rot_triplet_feats_clip.norm(dim=-1, keepdim=True)
            
                text_align_loss = F.l1_loss(triplet_feats_clip, clip_plabels)+F.l1_loss(rot_triplet_feats_clip, clip_plabels)
                
                if self.model.epoch>=startEpoch and self.isInit==True:
                    triplet_64_batch = torch.cat([triplet_feats_DEC, rot_triplet_feats_DEC], dim=0)

                    # 2. 计算 DEC Loss
                    # 得到当前batch的soft assignment q
                    dists1 = torch.sum((triplet_feats_DEC.unsqueeze(1) - self.model.cluster_centers) ** 2, dim=2)
                    q_batch = (1.0 + dists1).pow_(-1)
                    q_batch = (q_batch.T / torch.sum(q_batch, dim=1)).T
                    
                    dists2 = torch.sum((rot_triplet_feats_DEC.unsqueeze(1) - self.model.cluster_centers) ** 2, dim=2)
                    q_batch_rot = (1.0 + dists2).pow_(-1)
                    q_batch_rot = (q_batch_rot.T / torch.sum(q_batch_rot, dim=1)).T
                    
                    current_batch_size = q_batch.shape[0]
                    start_idx = sum(self.sample_counts[:batch_idx])  # 前batch_idx个batch的总样本数
                    end_idx = start_idx + current_batch_size
                    
                    # 获取对应的目标分布p
                    p_batch = self.p[start_idx : end_idx]
                    rot_p_batch = self.rot_p[start_idx : end_idx]
                    
                    dec_target_p = torch.cat([rot_p_batch,p_batch], dim=0)
                    q_batch = torch.cat([q_batch, q_batch_rot], dim=0)
                    
                    loss_dec = dec_loss(q_batch, dec_target_p)

                    total_loss = self_loss_edge + self_loss_obj + self_loss_triplet + text_align_loss + 5*loss_dec
                    loss_dict = {
                            "Epoch": self.model.epoch,
                            "Batch": batch_idx,
                            "Loss_obj": self_loss_obj.item(),
                            "Loss_edge": self_loss_edge.item(),
                            "Loss_triplet": self_loss_triplet.item(),
                            "text_align_loss": text_align_loss.item(),
                            "Loss_dec": loss_dec.item(),
                            "Total_loss": total_loss.item()
                        }
                else:
                    total_loss = self_loss_edge+self_loss_obj+self_loss_triplet+text_align_loss 
                    loss_dict = {
                            "Epoch": self.model.epoch,
                            "Batch": batch_idx,
                            "Loss_obj": self_loss_obj.item(),
                            "Loss_edge": self_loss_edge.item(),
                            "Loss_triplet": self_loss_triplet.item(),
                            "lr": self.model.optimizer.param_groups[0]['lr'],
                            "text_align_loss": text_align_loss.item(),
                            "Total_loss": total_loss.item()
                        }

                self.model.backward(total_loss)

                
                # 打印所有损失（一行显示）
                print(", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()]))

                if self.model.iteration >= self.max_iteration:
                    break
            
            loader = iter(train_loader)

            if (self.model.epoch+1) % 5 == 0:

                torch.save(self.model.mmg.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_mmg.pth")
                torch.save(self.model.obj_encoder.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(
                               self.model.epoch) + "_obj_encoder.pth")
                torch.save(self.model.rel_encoder_3d.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(
                               self.model.epoch) + "_rel_encoder.pth")
                torch.save(self.model.mlp_3d.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_mlp.pth")
                torch.save(self.model.mlp_obj_forBT.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_mlp_obj_forBT.pth")
                torch.save(self.model.mlp_edge_forBT.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_mlp_edge_forBT.pth")
                torch.save(self.model.mlp_triplet_forBT.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_mlp_triplet_forBT.pth")
                torch.save(self.model.triplet_projector_3d_forCLIP.state_dict(),
                           "/home/honsen/tartan/msg_data/pretrained_param_trans/epoch_" + str(self.model.epoch) + "_triplet_projector_3d_forCLIP.pth")
              
            self.model.epoch += 1


                   
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def save(self,epoch):
        self.model.save(epoch)

    
