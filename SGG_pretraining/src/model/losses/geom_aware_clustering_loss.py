import torch
import torch.nn as nn
import torch.nn.functional as F
# import faiss
import numpy as np
from sklearn.cluster import KMeans

class GeometricAwareClustering(nn.Module):
    def __init__(self, geo_dim=11, feat_dim=128, init_K=10, merge_thresh=0.3, split_thresh=0.7):
        super().__init__()
        # 几何原型与语义原型分开存储
        self.geo_prototypes = nn.Parameter(torch.randn(init_K, geo_dim))  # [K,11]
        self.feat_prototypes = nn.Parameter(torch.randn(init_K, feat_dim))  # [K,128]
        self.K = init_K
        self.merge_thresh = merge_thresh
        self.split_thresh = split_thresh
        
    def forward(self, geo_feats, sem_feats):
        """ 返回语义特征的软分配概率 """
        # L2归一化保证距离度量合理
        geo_feats = F.normalize(geo_feats, p=2, dim=1)  # [N,11]
        sem_feats = F.normalize(sem_feats, p=2, dim=1)  # [N,128]
        geo_centers = F.normalize(self.geo_prototypes, p=2, dim=1)
        feat_centers = F.normalize(self.feat_prototypes, p=2, dim=1)
        
        # 计算几何-语义联合距离
        geo_dist = torch.cdist(geo_feats, geo_centers)  # [N,K]
        sem_dist = torch.cdist(sem_feats, feat_centers)  # [N,K]
        joint_dist = 0.7*sem_dist + 0.3*geo_dist  # 加权融合
        
        # 软分配概率 (温度系数τ=0.1)
        p = F.softmax(-joint_dist/0.1, dim=1)  # [N,K]
        return p
    
    def dynamic_adjust(self, geo_feats, sem_feats):
        """ 每100次迭代动态调整簇 """
        with torch.no_grad():
            # 获取当前分配
            p = self.forward(geo_feats, sem_feats)
            assignments = p.argmax(dim=1)  # [N]
            
            # ---- 合并检查 ----
            center_dists = torch.cdist(
                torch.cat([self.geo_prototypes, self.feat_prototypes], dim=1),
                torch.cat([self.geo_prototypes, self.feat_prototypes], dim=1)
            )
            merge_mask = (center_dists < self.merge_thresh).triu(diagonal=1)
            
            # 执行合并
            merged_geo = []
            merged_feat = []
            used = set()
            for k in range(self.K):
                if k not in used:
                    to_merge = torch.where(merge_mask[k])[0]
                    if len(to_merge) > 0:
                        new_geo = self.geo_prototypes[[k]+to_merge.tolist()].mean(0)
                        new_feat = self.feat_prototypes[[k]+to_merge.tolist()].mean(0)
                        merged_geo.append(new_geo)
                        merged_feat.append(new_feat)
                        used.update(to_merge.tolist())
                    else:
                        merged_geo.append(self.geo_prototypes[k])
                        merged_feat.append(self.feat_prototypes[k])
            
            # ---- 分裂检查 ----
            new_geo_centers = []
            new_feat_centers = []
            for k in range(len(merged_geo)):
                mask = (assignments == k)
                if mask.sum() > 10:  # 簇足够大才考虑分裂
                    intra_dist = torch.cdist(sem_feats[mask], merged_feat[k].unsqueeze(0)).mean()
                    if intra_dist > self.split_thresh:
                        # 在该簇内做二次K-means
                        sub_feats = sem_feats[mask]
                        ksub = 2
                        kmeans = KMeans(n_clusters=ksub).fit(sub_feats.cpu().numpy())
                        new_feat_centers.extend(torch.from_numpy(kmeans.cluster_centers_).to(sem_feats))
                        # 几何中心按样本比例分配
                        geo_weights = p[mask, k]
                        for i in range(ksub):
                            sub_mask = (kmeans.labels_ == i)
                            new_geo_centers.append(
                                (geo_feats[mask][sub_mask] * geo_weights[sub_mask][:,None]).sum(0) / 
                                geo_weights[sub_mask].sum())
                    else:
                        new_geo_centers.append(merged_geo[k])
                        new_feat_centers.append(merged_feat[k])
                else:
                    new_geo_centers.append(merged_geo[k])
                    new_feat_centers.append(merged_feat[k])
            
            # 更新原型和K值
            self.K = len(new_geo_centers)
            self.geo_prototypes = nn.Parameter(torch.stack(new_geo_centers))
            self.feat_prototypes = nn.Parameter(torch.stack(new_feat_centers))

def cluster_consistency_loss(z_a, z_b, geo_a, geo_b, cluster_module):
    """
    z_a, z_b: 双视图的语义特征 [N,128]
    geo_a, geo_b: 对应的几何特征 [N,11]
    cluster_module: GeometricAwareClustering实例
    """
    # 获取两个视图的软分配概率
    p_a = cluster_module(geo_a, z_a)  # [N,K]
    p_b = cluster_module(geo_b, z_b)  # [N,K]
    
    # 对称KL散度损失
    loss = 0.5 * (F.kl_div(p_a.log(), p_b, reduction='batchmean') + 
                 F.kl_div(p_b.log(), p_a, reduction='batchmean'))
    
    # 防止平凡解：鼓励平均分配熵最大化
    avg_p = 0.5 * (p_a.mean(0) + p_b.mean(0))  # [K]
    entropy_loss = torch.sum(avg_p * torch.log(avg_p + 1e-10))  # 最大化熵=最小化负熵
    
    return loss + 0.1 * entropy_loss         

if __name__ =="__main__":
    za = torch.randn((1024,128))
    zb = torch.randn((1024, 128))

    geo_a = torch.randn((1024,11))
    geo_b = torch.randn((1024, 11))

    cluster_module = GeometricAwareClustering()

    loss = cluster_consistency_loss(za,zb,geo_a,geo_b,cluster_module)

    print()

class BarlowTwinsWithClustering(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.encoder = EdgeEncoder(feat_dim)  # 你的边特征编码器
        self.projector = nn.Linear(feat_dim, feat_dim)
        self.cluster = GeometricAwareClustering()
        
    def forward(self, batch):
        # 获取双视图数据
        x_a, x_b = batch['view1'], batch['view2']  # 各包含geo_feats和原始边特征
        geo_a, geo_b = x_a['geo'], x_b['geo']
        
        # 特征提取
        z_a = self.projector(self.encoder(x_a['feats']))  # [N,128]
        z_b = self.projector(self.encoder(x_b['feats']))
        
        # Barlow Twins损失
        bt_loss = barlow_twins_loss(z_a, z_b)  # 标准BT实现
        
        # 聚类一致性损失
        if self.training and (self.global_step % 100 == 0):
            self.cluster.dynamic_adjust(
                torch.cat([geo_a, geo_b]), 
                torch.cat([z_a.detach(), z_b.detach()])
            )
        cluster_loss = cluster_consistency_loss(z_a, z_b, geo_a, geo_b, self.cluster)
        
        return bt_loss + 0.3 * cluster_loss  # 加权组合