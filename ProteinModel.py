import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from Transformer import TransformerEncoderLayer, TransformerDecoderLayer


class ProteinGNNTransformer(nn.Module):
    def __init__(self, num_proteins, num_amino_acids, num_properties, latent_dim):
        super(ProteinGNNTransformer, self).__init__()

        # 蛋白质、氨基酸和属性的嵌入
        self.protein_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_proteins, latent_dim)))
        self.aa_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_amino_acids, latent_dim)))
        self.property_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_properties, latent_dim)))

        # Transformer编码器
        self.transformer_encoder = TransformerEncoderLayer(d_model=latent_dim, num_heads=args.num_head,
                                                           dropout=args.dropout)

        # 解码器组件
        self.decoder = TransformerDecoderLayer(d_model=latent_dim, num_heads=args.num_head, dropout=args.dropout)
        self.output_layer = nn.Linear(latent_dim, num_amino_acids)  # 输出层用于生成氨基酸序列

    def gnn_message_passing(self, adj, embeds):
        # 图消息传递函数保持不变
        return torch.spmm(adj, embeds)

    def forward(self, adj):
        # 合并所有嵌入
        embeds = [torch.concat([self.protein_embedding, self.aa_embedding, self.property_embedding], dim=0)]

        # 迭代更新嵌入
        for i in range(args.block_num):
            tmp_embeds = self.gnn_message_passing(adj, embeds[-1])

            # 通过Transformer更新
            tmp_embeds = self.transformer_encoder(tmp_embeds.unsqueeze(0)).squeeze(0)

            # 残差连接
            tmp_embeds += embeds[-1]
            embeds.append(tmp_embeds)

        # 聚合所有嵌入
        embeds = sum(embeds)

        # 分离不同类型的嵌入
        protein_embeds = embeds[:num_proteins]
        aa_embeds = embeds[num_proteins:num_proteins + num_amino_acids]
        prop_embeds = embeds[num_proteins + num_amino_acids:]

        return embeds, protein_embeds, aa_embeds, prop_embeds

    def decode_sequence(self, encoder_output, target_seq=None, max_length=100):
        # 解码蛋白质序列
        batch_size = encoder_output.size(0)

        if target_seq is not None:
            # 训练模式 - 使用teacher forcing
            decoder_output = self.decoder(target_seq, encoder_output)
            logits = self.output_layer(decoder_output)
            return logits
        else:
            # 推理模式 - 自回归生成
            # 实现序列生成逻辑
            pass