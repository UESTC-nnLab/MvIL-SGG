import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.losses.geom_aware_clustering_loss import cluster_consistency_loss, GeometricAwareClustering
# -----------------
# Helper Functions
# -----------------

# 损失函数2: DEC Loss (KL散度)
def dec_loss(q, p):
    # q: soft assignment, p: target distribution
    return nn.KLDivLoss(reduction='batchmean')(q.log(), p)

def off_diagonal(x):
    """
    Returns a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def calculate_barlow_twins_loss(z_a, z_b, lambd=5e-3):
    """
    Computes the Barlow Twins loss.
    """
    # Normalize the projector outputs
    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
    z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)

    batch_size, feature_dim = z_a.shape
    
    # Cross-correlation matrix
    c = torch.matmul(z_a_norm.T, z_b_norm) / batch_size

    # Invariance term (encourage diagonal elements to be 1)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    
    # Redundancy reduction term (encourage off-diagonal elements to be 0)
    off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + lambd * off_diag
    return loss

def calculate_coding_rate_loss(Z):
    """
    Computes the coding rate regularization loss (a simplified version).
    This version encourages the features to have a smaller variance, 
    effectively compressing the information.
    A more formal way is to penalize the log-determinant of the covariance matrix.
    """
    # Center the features
    Z_centered = Z - Z.mean(dim=0, keepdim=True)

    m = (len(Z))
    eps = 1e-2
    # Covariance matrix
    covariance = torch.matmul(Z_centered.T, Z_centered)# / (m*eps)
    
    # The coding rate loss is approximated by the log-determinant of the covariance
    # To ensure the matrix is positive definite for logdet, we add a small epsilon.

    scalar = covariance.shape[0] / (m*eps)
    covariance = covariance*scalar + torch.eye(covariance.shape[0], device=covariance.device) #* eps

    # log_det is a way to measure the "volume" of the feature space
    # Penalizing it encourages the model to compress the features.
    log_det = torch.logdet(covariance)
    
    # We want to minimize this, so the loss is -log_det or just the log_det
    # depending on the formulation. Let's assume we want to minimize the volume.
    return -1/2*log_det

# -----------------
# Main Model and Training Step
# -----------------

class BarlowTwinsWithCodingRate1(nn.Module):
    def __init__(self, backbone, projector_dims, lambd=5e-3, gamma=1e-2):
        super().__init__()
        self.backbone = backbone
        self.lambd = lambd  # Barlow Twins lambda
        self.gamma = gamma  # Coding rate gamma
        
        # Projector
        sizes = [backbone.output_dim] + projector_dims
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, x_a, x_b):
        # Get representations
        h_a = self.backbone(x_a)
        h_b = self.backbone(x_b)
        
        # Project to the embedding space
        z_a = self.projector(h_a)
        z_b = self.projector(h_b)
        
        # Calculate Barlow Twins loss
        bt_loss = calculate_barlow_twins_loss(z_a, z_b, self.lambd)
        
        # Calculate Coding Rate loss on the combined batch of projections
        # We can apply it to z_a, z_b, or both. Applying to both is common.
        z_combined = torch.cat([z_a, z_b], dim=0)
        cr_loss = calculate_coding_rate_loss(z_combined)

        asd = self.gamma * cr_loss
        # Total loss
        total_loss = bt_loss + self.gamma * cr_loss
        
        return total_loss, bt_loss, cr_loss


class BarlowTwinsWithCodingRate(nn.Module):
    def __init__(self, lambd=5e-3, gamma=1e-3, isCR=True):
        super().__init__()
        self.lambd = lambd  # Barlow Twins lambda
        self.gamma = gamma  # Coding rate gamma
        self.isCR = isCR
    def forward(self, z_a, z_b):

        # Calculate Barlow Twins loss
        bt_loss = calculate_barlow_twins_loss(z_a, z_b, self.lambd)

        # Calculate Coding Rate loss on the combined batch of projections
        # We can apply it to z_a, z_b, or both. Applying to both is common.
        if self.isCR:
            z_combined = torch.cat([z_a, z_b], dim=0)
            cr_loss = calculate_coding_rate_loss(z_combined)

            # Total loss
            total_loss = bt_loss + self.gamma * cr_loss
        else:
            total_loss = bt_loss

        return total_loss

class BarlowTwins_loss(nn.Module):
    def __init__(self, lambd=5e-3):
        super().__init__()
        self.lambd = lambd  # Barlow Twins lambda
    def forward(self, z_a, z_b):

        # Calculate Barlow Twins loss
        bt_loss = calculate_barlow_twins_loss(z_a, z_b, self.lambd)

        return bt_loss

class BarlowTwins_CR_GeomAwareClustering(nn.Module):
    def __init__(self, lambd=5e-3, gamma=1e-3, GAcluster_dims=512*3):
        super().__init__()
        self.lambd = lambd  # Barlow Twins lambda
        self.gamma = gamma  # Coding rate gamma
        self.GACluster = GeometricAwareClustering(feat_dim=GAcluster_dims)

    def forward(self, z_a, z_b, geom_a, geom_b, global_step):

        # Calculate Barlow Twins loss
        bt_loss = calculate_barlow_twins_loss(z_a, z_b, self.lambd)

        # Calculate Coding Rate loss on the combined batch of projections
        # We can apply it to z_a, z_b, or both. Applying to both is common.
        z_combined = torch.cat([z_a, z_b], dim=0)
        cr_loss = calculate_coding_rate_loss(z_combined)

        # 聚类一致性损失
        if global_step % 100 == 0:
            self.GACluster.dynamic_adjust(
                torch.cat([geom_a, geom_b]),
                torch.cat([z_a.detach(), z_b.detach()])
            )

        cluster_loss = cluster_consistency_loss(z_a, z_b, geom_a, geom_b, self.GACluster)

        # Total loss
        total_loss = bt_loss + self.gamma * cr_loss

        return total_loss, bt_loss, cr_loss

class BarlowTwins_CR_DEC(nn.Module):
    def __init__(self, lambd=5e-3, gamma=1e-3, GAcluster_dims=512*3):
        super().__init__()
        self.lambd = lambd  # Barlow Twins lambda
        self.gamma = gamma  # Coding rate gamma
        self.GACluster = GeometricAwareClustering(feat_dim=GAcluster_dims)

    def forward(self, z_a, z_b, geom_a, geom_b, global_step):

        # Calculate Barlow Twins loss
        bt_loss = calculate_barlow_twins_loss(z_a, z_b, self.lambd)

        # Calculate Coding Rate loss on the combined batch of projections
        # We can apply it to z_a, z_b, or both. Applying to both is common.
        z_combined = torch.cat([z_a, z_b], dim=0)
        cr_loss = calculate_coding_rate_loss(z_combined)

        # 聚类一致性损失
        if global_step % 100 == 0:
            self.GACluster.dynamic_adjust(
                torch.cat([geom_a, geom_b]),
                torch.cat([z_a.detach(), z_b.detach()])
            )

        cluster_loss = cluster_consistency_loss(z_a, z_b, geom_a, geom_b, self.GACluster)

        # Total loss
        total_loss = bt_loss + self.gamma * cr_loss

        return total_loss, bt_loss, cr_loss

# -----------------
# Example Usage
# -----------------
if __name__ == '__main__':
    # Dummy backbone (e.g., a ResNet)
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 512)
            self.output_dim = 512
        
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    # --- Configuration ---
    batch_size = 128
    projector_dims = [2048, 2048, 2048]
    barlow_lambda = 5e-3
    coding_rate_gamma = 1e-2 # This needs careful tuning

    # --- Model Initialization ---
    backbone = DummyBackbone()
    model = BarlowTwinsWithCodingRate(backbone, projector_dims, lambd=barlow_lambda, gamma=coding_rate_gamma)
    
    # --- Dummy Data ---
    # Two augmented views of the same batch of images
    dummy_x_a = torch.randn(batch_size, 3, 32, 32)
    dummy_x_b = torch.randn(batch_size, 3, 32, 32)

    # --- Optimizer ---
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # --- Training Step ---
    model.train()
    optimizer.zero_grad()
    
    total_loss, bt_loss, cr_loss = model(dummy_x_a, dummy_x_b)
    
    total_loss.backward()
    optimizer.step()
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  - Barlow Twins Loss: {bt_loss.item():.4f}")
    print(f"  - Coding Rate Loss: {cr_loss.item():.4f}")