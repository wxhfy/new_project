import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter
from tqdm import tqdm


class EdgeAttention(nn.Module):
    """边特征注意力模块，针对肽键和空间连接赋予不同权重"""

    def __init__(self, edge_dim):
        super(EdgeAttention, self).__init__()
        self.edge_proj = nn.Linear(edge_dim, 1)

    def forward(self, edge_attr):
        # 计算边的注意力权重
        edge_weights = torch.sigmoid(self.edge_proj(edge_attr))
        return edge_weights


class ProteinGATLayer(nn.Module):
    """蛋白质知识图谱专用GAT层，包含边特征和残差连接"""

    def __init__(self,
                 in_dim,
                 out_dim,
                 heads=4,
                 dropout=0.1,
                 edge_dim=2,
                 use_residual=True,
                 use_edge_features=True,
                 use_gatv2=True):
        super(ProteinGATLayer, self).__init__()

        # 选择GAT版本 (GATv2改进了原始GAT的表达能力)
        if use_gatv2:
            self.gat = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim // heads,  # 多头注意力的输出维度
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim if use_edge_features else None,
                add_self_loops=True,  # 添加自环以聚合自身特征
                concat=True
            )
        else:
            self.gat = GATConv(
                in_channels=in_dim,
                out_channels=out_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim if use_edge_features else None,
                add_self_loops=True,
                concat=True
            )

        # 图归一化层 - 帮助稳定训练
        self.norm = GraphNorm(out_dim)

        # 额外的边特征注意力
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.edge_attn = EdgeAttention(edge_dim)

        # 残差连接处理
        self.use_residual = use_residual
        if use_residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 边特征处理
        edge_weights = None
        if self.use_edge_features and edge_attr is not None:
            edge_weights = self.edge_attn(edge_attr)
            # 根据边类型调整边特征 (肽键vs空间连接)
            edge_type = edge_attr[:, 0].long()  # 假设第一列是边类型 (0=空间, 1=肽键)
            peptide_mask = (edge_type == 1).float().view(-1, 1)
            # 增强肽键的权重 - 基于生物学知识
            edge_weights = edge_weights * (1.0 + 0.5 * peptide_mask)

        # GAT主处理
        out = self.gat(x, edge_index, edge_attr=edge_attr)

        # 残差连接
        if self.use_residual:
            res = self.res_proj(x)
            out = out + res

        # 归一化和非线性激活
        out = self.norm(out, batch)
        out = F.elu(out)

        return out


class AminoAcidEmbedding(nn.Module):
    """氨基酸特征嵌入层，将离散和连续特征转换为统一表示"""

    def __init__(self, embed_dim=64, use_position=True):
        super(AminoAcidEmbedding, self).__init__()

        # 氨基酸类型嵌入
        self.aa_embedding = nn.Embedding(20, embed_dim // 4)  # 20种标准氨基酸

        # 物理化学性质编码器
        self.physchem_encoder = nn.Sequential(
            nn.Linear(6, embed_dim // 4),  # 疏水性、电荷、极性、分子量、二级结构等
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim // 4)
        )

        # 结构信息编码器
        self.struct_encoder = nn.Sequential(
            nn.Linear(4, embed_dim // 4),  # 二级结构、可接触面积、phi、psi角度
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim // 4)
        )

        # 位置编码器 (相对于序列或空间)
        self.use_position = use_position
        if use_position:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, embed_dim // 4),  # 3D坐标
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim // 4)
            )

        # 特征整合
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, aa_indices, physchem_feats, struct_feats, pos_feats=None):
        # 氨基酸类型嵌入
        aa_emb = self.aa_embedding(aa_indices)

        # 物理化学性质编码
        phys_emb = self.physchem_encoder(physchem_feats)

        # 结构信息编码
        struct_emb = self.struct_encoder(struct_feats)

        # 拼接特征
        if self.use_position and pos_feats is not None:
            pos_emb = self.pos_encoder(pos_feats)
            combined = torch.cat([aa_emb, phys_emb, struct_emb, pos_emb], dim=1)
        else:
            combined = torch.cat([aa_emb, phys_emb, struct_emb], dim=1)

        # 投影到输出维度
        output = self.output_proj(combined)
        return output


class ProteinGATModel(nn.Module):
    """蛋白质知识图谱GAT嵌入模型"""

    def __init__(self,
                 node_features=7,  # 默认节点特征数量
                 edge_features=2,  # 边特征数量 [edge_type, distance]
                 hidden_dim=128,  # 隐藏层维度
                 output_dim=64,  # 输出嵌入维度
                 num_layers=3,  # GAT层数量
                 heads=4,  # 注意力头数量
                 dropout=0.1,  # Dropout比例
                 residual=True,  # 是否使用残差连接
                 pooling='attention',  # 池化方法: 'mean', 'max', 'add', 'attention'
                 use_edge_features=True,  # 是否使用边特征
                 use_gatv2=True):  # 是否使用GATv2
        super(ProteinGATModel, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.use_edge_features = use_edge_features

        # 节点输入特征映射层
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # GAT层堆叠
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            self.gat_layers.append(
                ProteinGATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_features if use_edge_features else None,
                    use_residual=residual,
                    use_edge_features=use_edge_features,
                    use_gatv2=use_gatv2
                )
            )

        # 注意力池化层 (如果使用)
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )

        # 输出映射层
        self.output_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """
        前向传播函数

        参数:
            data: PyG数据对象，包含:
                - x: 节点特征 [num_nodes, node_features]
                - edge_index: 边连接 [2, num_edges]
                - edge_attr: 边特征 [num_edges, edge_features]
                - batch: 批处理索引 [num_nodes]

        返回:
            graph_emb: 图嵌入 [batch_size, output_dim]
            node_emb: 节点嵌入 [num_nodes, output_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 如果batch未提供，假设只有一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 节点特征编码
        x = self.node_encoder(x)

        # 通过GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr, batch)

        # 保存节点嵌入结果
        node_embeddings = x

        # 图级池化
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(x, batch)
        elif self.pooling == 'add':
            graph_embedding = global_add_pool(x, batch)
        elif self.pooling == 'attention':
            # 计算注意力分数
            attn_scores = self.attention_pool(x)
            attn_weights = F.softmax(attn_scores, dim=0)
            # 加权聚合
            graph_embedding = scatter(x * attn_weights, batch, dim=0, reduce='sum')

        # 映射到最终嵌入维度
        graph_embedding = self.output_mapper(graph_embedding)

        return graph_embedding, node_embeddings

    def preprocess_protein_graph(self, data):
        """
        预处理蛋白质图谱数据，从原始节点特征中提取结构化特征

        参数:
            data: 包含蛋白质图谱特征的PyG数据对象

        返回:
            处理后的特征
        """
        x = data.x  # 原始特征

        # 从知识图谱中提取关键特征
        # 假设特征顺序是 [hydropathy, charge, plddt, ...]
        hydropathy = x[:, 0].view(-1, 1)  # 疏水性
        charge = x[:, 1].view(-1, 1)  # 电荷
        plddt = x[:, 2].view(-1, 1)  # 预测质量

        # 计算额外的特征（例如可以添加其他特性）
        # 这里我们可以加入一些基于AMPs的专门特征

        # 例如，计算疏水矩（量化两亲性）
        amphipathicity = torch.abs(hydropathy) * (1 + torch.abs(charge))

        # 构建增强特征向量
        enhanced_features = torch.cat([
            x,  # 原始特征
            amphipathicity,  # 两亲性指标
            hydropathy * charge,  # 疏水-电荷交互
            plddt * hydropathy,  # 质量加权疏水性
        ], dim=1)

        # 更新数据对象
        data.x = enhanced_features

        return data


    def embed_protein_graphs(graph_data_list, model, device='cuda', batch_size=32):
        """
        嵌入一组蛋白质图谱

        参数:
            graph_data_list: 蛋白质图谱数据列表(PyG格式)
            model: 训练好的ProteinGATModel
            device: 计算设备
            batch_size: 批处理大小

        返回:
            embeddings: 字典，映射图谱ID到嵌入向量
        """
        model.eval()
        model = model.to(device)
        embeddings = {}

        # 批量处理
        loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(loader, desc="嵌入蛋白质图谱"):
                batch = batch.to(device)

                # 预处理和嵌入
                batch = model.preprocess_protein_graph(batch)
                graph_emb, _ = model(batch)

                # 保存结果
                for i, graph_id in enumerate(batch.graph_id):
                    embeddings[graph_id] = graph_emb[i].cpu().numpy()

        return embeddings


    def train_protein_gat_model(train_data, val_data, model_params, train_params):
        """
        训练蛋白质GAT模型

        参数:
            train_data: 训练数据
            val_data: 验证数据
            model_params: 模型参数字典
            train_params: 训练参数字典

        返回:
            训练好的模型
        """
        # 初始化模型
        model = ProteinGATModel(**model_params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 数据加载器
        train_loader = DataLoader(train_data, batch_size=train_params['batch_size'],
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=train_params['batch_size'],
                                shuffle=False, num_workers=4)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'],
                                     weight_decay=train_params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=5)

        # 自监督对比损失函数
        def contrastive_loss(embeddings, batch, tau=0.1):
            # 计算余弦相似度矩阵
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / tau

            # 创建标签：相同批次的是正例，不同批次的是负例
            labels = torch.eq(batch.view(-1, 1), batch.view(1, -1)).float()

            # 对角线上是自身，排除
            mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
            sim_matrix = sim_matrix * (1 - mask) - 1e9 * mask

            # 对比损失计算
            pos_sim = torch.exp(sim_matrix) * labels
            neg_sim = torch.exp(sim_matrix) * (1 - labels)

            loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
            return loss.mean()

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(train_params['epochs']):
            # 训练阶段
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)

                # 预处理特征
                batch = model.preprocess_protein_graph(batch)

                # 前向传播
                graph_emb, node_emb = model(batch)

                # 计算损失 (自监督对比损失)
                loss = contrastive_loss(graph_emb, batch.batch)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    batch = model.preprocess_protein_graph(batch)
                    graph_emb, _ = model(batch)
                    loss = contrastive_loss(graph_emb, batch.batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 学习率调整
            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_protein_gat_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= train_params['patience']:
                    print(f"早停触发! 轮次 {epoch}")
                    break

            print(f"轮次 {epoch + 1}/{train_params['epochs']}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 加载最佳模型
        model.load_state_dict(torch.load('best_protein_gat_model.pt'))
        return model


