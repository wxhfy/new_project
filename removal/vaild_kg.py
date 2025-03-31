# 验证PyG格式数据
import torch
from torch_geometric.data import Data

# 加载图谱数据
data_path = "/home/20T-1/fyh0106/kg/ALL/batch_1/knowledge_graphs_pyg/protein_kg_chunk_1.pt"
graph_data = torch.load(data_path)

# 检查第一个图谱
first_key = list(graph_data.keys())[1]
first_graph = graph_data[first_key]

# 打印图谱信息
print(f"图谱ID: {first_key}")
print(f"节点数: {first_graph.x.shape[0]}")
print(f"边数: {first_graph.edge_index.shape[1]}")
print(f"节点特征维度: {first_graph.x.shape[1]}")
print(f"边特征维度: {first_graph.edge_attr.shape[1] if first_graph.edge_attr is not None else 0}")
print(f"蛋白质序列: {first_graph.sequence}")