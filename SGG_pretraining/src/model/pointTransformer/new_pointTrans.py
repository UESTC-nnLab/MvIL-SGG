import torch
import torch.nn as nn
import einops
from src.model.pointTransformer.cpp.pointops.functions import pointops  # 需要安装 pointops 库: pip install pointops

# ==============================================================================
# 提供的原始模块 (PointTransformerLayer, TransitionDown, Bottleneck)
# 以下代码来自您提供的脚本，为了完整性在此列出
# ==============================================================================

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1,
                                                                                                          2).contiguous() if i == 1 else layer(
            p_r)  # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1,
                                                                                                      2).contiguous() if i % 3 == 0 else layer(
            w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


# ==============================================================================
# Part 2: 构建 Point Transformer 编码器
# ==============================================================================

class PointTransformerEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, block=PointTransformerBlock, blocks_config=[1, 1, 1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # ------ 1. 编码器骨干网络 (Encoder Backbone) ------
        # 这部分结构直接取自 PointTransformerSeg 的 encoder 部分
        self.in_planes = in_channels
        planes = [32, 64, 128, 512, 512]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        
        self.enc1 = self._make_enc(block, planes[0], blocks_config[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks_config[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks_config[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks_config[3], share_planes, stride[3], nsample[3])
        # self.enc5 = self._make_enc(block, planes[4], blocks_config[4], share_planes, stride[4], nsample[4])
        
        # ------ 2. 全局特征聚合与MLP头 (Aggregation and MLP Head) ------
        self.mlp_head = nn.Sequential(
            nn.Linear(planes[3], planes[3]),
            nn.BatchNorm1d(planes[3]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[3], latent_dim)
        )

    def _make_enc(self, block, planes, blocks, share_planes, stride, nsample):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
    
    def _convert_bcn_to_pt_format(self, features_bcn):
        """将 (B, C, N) 格式转换为 (coord, feat, offset) 格式"""
        B, C, N = features_bcn.shape
        device = features_bcn.device

        # 假设前3个通道是坐标，其余是特征
        coord_b3n = features_bcn[:, :3, :]
        
        # (B, 3, N) -> (B, N, 3) -> (B*N, 3)
        coord = coord_b3n.permute(0, 2, 1).reshape(-1, 3)
        
        # (B, C, N) -> (B, N, C) -> (B*N, C)
        feat = features_bcn.permute(0, 2, 1).reshape(-1, self.in_channels)
        
        # 创建 offset
        offset = torch.arange(1, B + 1, device=device).int() * N
        
        return coord, feat, offset

    def forward(self, x_bcn):
        """
        输入:
            x_bcn: (B, C, N) 格式的点云张量。C 必须与 in_channels 匹配。
        输出:
            embedding: (B, D) 格式的全局特征嵌入。D=latent_dim。
        """
        # --- 1. 输入格式转换 ---
        p0, x0, o0 = self._convert_bcn_to_pt_format(x_bcn)
        
        # --- 2. 通过编码器骨干网络 ---
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        # p5, x5, o5 = self.enc5([p4, x4, o4])
        # x5 是最深层、最抽象的逐点特征, shape: (M_5, 512)
        # o5 是对应的 offset, shape: (B,)
        
        # --- 3. 全局特征聚合 (Batched Pooling) ---
        # 我们对每个点云的特征进行平均池化
        batch_size = o4.shape[0]
        pooled_features = []
        start_index = 0
        for i in range(batch_size):
            end_index = o4[i]
            # 提取属于当前点云的特征
            sample_features = x4[start_index:end_index]
            # 对当前点云的特征求平均值，得到一个全局向量
            pooled_feature = sample_features.mean(dim=0)
            pooled_features.append(pooled_feature)
            start_index = end_index
        
        # 将列表中的全局向量堆叠成一个批次
        global_feature = torch.stack(pooled_features, dim=0) # (B, 512)
        
        # --- 4. 通过MLP头得到最终嵌入 ---
        embedding = self.mlp_head(global_feature) # (B, D)
        
        return embedding

# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == '__main__':
    # --- 参数 ---
    batch_size = 512
    num_points = 128
    in_channels = 3  # 假设输入是 XYZ + RGB (3+3)
    latent_dim = 768 # 目标嵌入维度

    # --- 实例化模型 ---
    # 使用 PointTransformerSeg26 类似的配置: [1, 1, 1, 1, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = PointTransformerEncoder(
        in_channels=in_channels, 
        latent_dim=latent_dim,
        blocks_config=[1, 1, 1, 1, 1] # 对应 PointTransformerSeg26
    ).to(device)

    # --- 创建模拟输入 ---
    # (B, C, N)
    mock_input = torch.randn(batch_size, in_channels, num_points).to(device)
    print(f"\nInput shape (B, C, N): {mock_input.shape}")

    # --- 前向传播 ---
    output_embedding = encoder(mock_input)

    # --- 打印输出 ---
    print(f"Output embedding shape (B, D): {output_embedding.shape}")
    
    # --- 验证输出维度 ---
    assert output_embedding.shape == (batch_size, latent_dim)
    print("\n✅  Successfully created an encoder to map (B, C, N) -> (B, D).")
    
    # --- 打印模型参数量 ---
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f} M")

