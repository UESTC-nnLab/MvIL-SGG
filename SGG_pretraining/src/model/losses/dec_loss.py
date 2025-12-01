import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans # 用于初始化聚类中心
import numpy as np

# --- 编码器 f(.) ---
class TripletEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(TripletEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        return self.feature_extractor(x)

# --- Barlow Twins Loss ---
def barlow_twins_loss(z_a, z_b, lambda_off_diag=0.005):
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)

    # 归一化 (Batch Normalization 也可以)
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-6)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-6)

    C = torch.matmul(z_a_norm.T, z_b_norm) / batch_size

    on_diag = torch.pow(1 - torch.diag(C), 2).sum()
    off_diag = (torch.sum(torch.triu(torch.pow(C, 2), diagonal=1)) +
                torch.sum(torch.tril(torch.pow(C, 2), diagonal=-1)))

    return on_diag + lambda_off_diag * off_diag

# --- DEC 聚类层和损失 ---
class DECLoss(nn.Module):
    def __init__(self, num_clusters, feature_dim, alpha=1.0):
        super(DECLoss, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.cluster_centers = None # nn.Parameter(torch.Tensor(num_clusters, feature_dim))
        # nn.init.xavier_uniform_(self.cluster_centers) # 初始化聚类中心

    def forward(self, features):
        """
        features: (batch_size, feature_dim)
        """
        # 计算 Q (软分配)
        # q_{ij} = (1 + ||z_i - m_j||^2 / alpha)^(-(alpha+1)/2)
        # dists: (batch_size, num_clusters)
        dists = torch.cdist(features, self.cluster_centers, p=2)**2
        q = 1.0 / (1.0 + dists / self.alpha)**((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True) # 归一化

        # 计算 P (目标分布)
        # p_{ij} = q_{ij}^2 / sum_{i'} q_{i'j}
        # p_{ij} = p_{ij} / sum_{k} p_{ik}
        p = q**2 / torch.sum(q, dim=0, keepdim=True) # 确保列和为1
        p = p / torch.sum(p, dim=1, keepdim=True) # 确保行和为1

        # KL 散度损失
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss, q # 返回损失和软分配Q，Q有时用于监控或后续处理

# --- 训练函数 ---
def train_bt_dec(model, dataloader, optimizer, num_epochs_pretrain, num_epochs_joint,
                 num_clusters, lambda_cluster, device):

    feature_dim = model.feature_extractor[-1].out_features
    dec_loss_fn = DECLoss(num_clusters, feature_dim).to(device)

    # --- 阶段 1: 预训练 Barlow Twins ---
    print("--- Phase 1: Pre-training with Barlow Twins ---")
    model.train()
    for epoch in range(num_epochs_pretrain):
        total_bt_loss = 0
        for batch_data in dataloader:
            triplet_view1 = batch_data['triplet_view1'].to(device)
            triplet_view2 = batch_data['triplet_view2'].to(device)

            optimizer.zero_grad()
            features_view1 = model(triplet_view1)
            features_view2 = model(triplet_view2)

            loss_bt = barlow_twins_loss(features_view1, features_view2)
            loss_bt.backward()
            optimizer.step()
            total_bt_loss += loss_bt.item()
        print(f"Pretrain Epoch {epoch+1}, Avg BT Loss: {total_bt_loss / len(dataloader):.4f}")

    # --- 阶段 2: 初始化聚类中心 ---
    print("--- Phase 2: Initializing Cluster Centers ---")
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch_data in dataloader:
            triplet_view1 = batch_data['triplet_view1'].to(device) # 使用一个视图的特征进行聚类
            features = model(triplet_view1)
            all_features.append(features.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)

    print("Running K-Means for initialization...")
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0) # n_init for robustness
    kmeans.fit(all_features)
    # 将K-Means得到的中心点赋值给DEC层的可学习参数
    dec_loss_fn.cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_).to(device))
    print("Cluster centers initialized.")

    # --- 阶段 3: 联合训练 Barlow Twins + DEC ---
    print("--- Phase 3: Joint Training with Barlow Twins and DEC ---")
    model.train()
    for epoch in range(num_epochs_joint):
        total_bt_loss = 0
        total_dec_loss = 0
        for batch_data in dataloader:
            triplet_view1 = batch_data['triplet_view1'].to(device)
            triplet_view2 = batch_data['triplet_view2'].to(device)

            optimizer.zero_grad()
            features_view1 = model(triplet_view1)
            features_view2 = model(triplet_view2)

            # Barlow Twins Loss
            loss_bt = barlow_twins_loss(features_view1, features_view2)

            # DEC Loss (使用features_view1进行DEC计算)
            # 注意：这里dec_loss_fn(features_view1) 会使得DEC的梯度回传到features_view1
            loss_dec, q_values = dec_loss_fn(features_view1)

            # 整体损失
            loss_total = loss_bt + lambda_cluster * loss_dec
            loss_total.backward()
            optimizer.step()

            total_bt_loss += loss_bt.item()
            total_dec_loss += loss_dec.item()

        print(f"Joint Epoch {epoch+1}, Avg BT Loss: {total_bt_loss / len(dataloader):.4f}, "
              f"Avg DEC Loss: {total_dec_loss / len(dataloader):.4f}")

    return model

# --- 示例使用 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_dim = 128 # 您的三元组特征维度
# feature_dim = 256 # 学习到的特征维度
# num_clusters = 10 # 期望的聚类数量 K

# model = TripletEncoder(input_dim, feature_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # 假设您有一个 DataLoader
# # dataloader = ... # 您的 DataLoader 应该返回 {'triplet_view1': ..., 'triplet_view2': ...}

# # 训练
# # train_bt_dec(model, dataloader, optimizer,
# #              num_epochs_pretrain=20,
# #              num_epochs_joint=50,
# #              num_clusters=num_clusters,
# #              lambda_cluster=0.5, # DEC 损失的权重，通常需要仔细调整
# #              device=device)

# # 训练完成后，您可以使用 model 提取特征，然后根据 dec_loss_fn.cluster_centers 进行聚类
# # 或者直接使用 q_values 进行软聚类分配

if __name__ == '__main__':
    decloss = DECLoss(10, 64)

    inps = torch.randn(1000, 64).cuda()
    inps1 = torch.randn(1000, 11).cuda()
    dists = torch.cdist(inps, inps1, p=2) ** 2

    decloss = decloss.cuda()
    loss, q = decloss(inps)
    print(loss, q.shape)