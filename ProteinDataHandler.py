import torch
import numpy as np
import pickle
import gzip
import json
from torch.utils.data import Dataset, DataLoader
import os
import time
from tqdm import tqdm


class ProteinDataset(Dataset):
    def __init__(self, data_dir, batch_size=1000, max_proteins=None, test_mode=False):
        """
        初始化蛋白质数据集

        参数:
            data_dir: 蛋白质数据目录
            batch_size: 批处理大小
            max_proteins: 最大蛋白质数量限制 (None表示不限制)
            test_mode: 测试模式，只加载少量数据用于验证
        """
        self.test_mode = test_mode
        if test_mode:
            print("⚠️ 运行在测试模式，只加载有限数量的蛋白质")
            max_proteins = max_proteins or 1000

        self.max_proteins = max_proteins
        self.data_dir = data_dir
        self.metadata = {}

        # 加载蛋白质数据
        print(f"正在加载蛋白质数据: {data_dir}")
        start_time = time.time()

        # 检查数据存储格式
        metadata_path = os.path.join(data_dir, "protein_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.format_type = self.metadata.get("format", "standard")
        else:
            print("⚠️ 未找到元数据文件，尝试推断数据格式...")
            if os.path.exists(os.path.join(data_dir, "protein_data.pkl.gz")):
                self.format_type = "compressed"
            elif os.path.exists(os.path.join(data_dir, "protein_index.pkl.gz")):
                self.format_type = "chunked_compressed"
            elif os.path.exists(os.path.join(data_dir, "protein_data.pkl")):
                self.format_type = "standard"
            else:
                raise FileNotFoundError(f"无法在 {data_dir} 找到可识别的蛋白质数据文件")

        print(f"数据格式: {self.format_type}")

        # 根据不同存储格式加载数据
        self.proteins = {}
        if self.format_type in ["chunked", "chunked_compressed"]:
            self._load_chunked_data()
        else:
            self._load_single_file()

        print(f"数据加载完成，用时: {time.time() - start_time:.2f}秒")

        # 处理限制
        if self.max_proteins and self.max_proteins < len(self.proteins):
            protein_ids = list(self.proteins.keys())[:self.max_proteins]
            self.proteins = {pid: self.proteins[pid] for pid in protein_ids}
            print(f"⚠️ 已限制为前 {self.max_proteins} 个蛋白质")

        # 提取并处理序列和属性
        self.protein_ids = list(self.proteins.keys())
        self.num_proteins = len(self.protein_ids)
        self.num_amino_acids = 20  # 20种标准氨基酸

        # 计算所有唯一的属性名
        all_props = set()
        for pid in self.protein_ids:
            props = self.proteins[pid].get('properties', {})
            if props:
                all_props.update(props.keys())

        self.num_properties = len(all_props)
        self.property_names = sorted(all_props)

        print(f"加载了 {self.num_proteins} 个蛋白质序列，{self.num_properties} 个属性")

    def _load_single_file(self):
        """加载单个文件格式的数据"""
        if self.format_type == "compressed":
            file_path = os.path.join(self.data_dir, "protein_data.pkl.gz")
            with gzip.open(file_path, 'rb') as f:
                self.proteins = pickle.load(f)
        else:
            file_path = os.path.join(self.data_dir, "protein_data.pkl")
            with open(file_path, 'rb') as f:
                self.proteins = pickle.load(f)

        print(f"从单个文件加载了 {len(self.proteins)} 个蛋白质")

    def _load_chunked_data(self):
        """加载分块数据"""
        # 加载索引
        index_path = os.path.join(self.data_dir, "protein_index.pkl.gz")
        if os.path.exists(index_path):
            with gzip.open(index_path, 'rb') as f:
                self.protein_index = pickle.load(f)
        else:
            raise FileNotFoundError(f"未找到索引文件: {index_path}")

        # 指定要加载的蛋白质数量
        all_protein_ids = list(self.protein_index.keys())
        if self.max_proteins:
            load_ids = all_protein_ids[:self.max_proteins]
            print(f"将只加载前 {len(load_ids)} 个蛋白质")
        else:
            load_ids = all_protein_ids

        # 按块ID组织蛋白质
        proteins_by_chunk = {}
        for pid in load_ids:
            chunk_id = self.protein_index[pid]
            if chunk_id not in proteins_by_chunk:
                proteins_by_chunk[chunk_id] = []
            proteins_by_chunk[chunk_id].append(pid)

        # 加载每个包含所需蛋白质的块
        print(f"需要加载 {len(proteins_by_chunk)} 个数据块...")

        for chunk_id, chunk_proteins in tqdm(proteins_by_chunk.items(), desc="加载数据块"):
            # 修正这里: 使用正确的文件名模式
            chunk_file = os.path.join(self.data_dir, f"protein_data_chunk_{chunk_id}.pkl.gz")

            try:
                with gzip.open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)

                # 只提取需要的蛋白质
                for pid in chunk_proteins:
                    if pid in chunk_data:
                        self.proteins[pid] = chunk_data[pid]
            except Exception as e:
                print(f"⚠️ 警告: 加载数据块 {chunk_id} 时出错: {str(e)}")

    def __len__(self):
        """返回数据集中蛋白质的数量"""
        return len(self.protein_ids)

    def __getitem__(self, idx):
        """返回指定索引的蛋白质数据"""
        # 获取蛋白质ID
        protein_id = self.protein_ids[idx]
        protein_data = self.proteins[protein_id]

        # 获取序列
        sequence = protein_data.get('sequence', '')

        # 获取属性
        properties = protein_data.get('properties', {})

        # 将序列转换为氨基酸索引序列
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20种标准氨基酸
        aa_indices = torch.tensor([amino_acids.find(aa) for aa in sequence if aa in amino_acids])

        # 将属性转换为特征向量
        property_vector = torch.zeros(len(self.property_names))

        for i, prop_name in enumerate(self.property_names):
            if prop_name in properties:
                property_vector[i] = float(properties[prop_name])

        return {
            'protein_id': protein_id,
            'sequence': aa_indices,
            'properties': property_vector,
            'raw_sequence': sequence
        }


class ProteinDataHandler:
    def __init__(self, data_dir, batch_size=32, test_mode=False, max_proteins=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_mode = test_mode

        print(f"初始化蛋白质数据处理器，批处理大小: {batch_size}")
        if test_mode:
            print("🧪 测试模式已启用，将只加载少量数据")
            max_proteins = max_proteins or 1000

        # 加载数据集
        self.dataset = ProteinDataset(
            data_dir,
            batch_size=1000,  # 内部批处理大小
            max_proteins=max_proteins,
            test_mode=test_mode
        )

        # 设置数据集属性
        self.num_proteins = self.dataset.num_proteins
        self.num_amino_acids = self.dataset.num_amino_acids
        self.num_properties = self.dataset.num_properties

        print(f"数据集初始化完成: {self.num_proteins}个蛋白质, {self.num_properties}个属性")

    def test_loading(self, num_samples=5):
        """测试数据集加载，显示几个样本"""
        print(f"\n=== 测试数据集加载 ({num_samples}个样本) ===")

        # 如果需要查看指定数量的样本
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            protein_id = sample['protein_id']
            seq_len = len(sample['sequence'])
            prop_count = (sample['properties'] != 0).sum().item()

            print(f"样本 {i}:")
            print(f"  - 蛋白质ID: {protein_id}")
            print(f"  - 序列长度: {seq_len}")
            print(f"  - 有值属性数量: {prop_count}")
            print(f"  - 序列前20个氨基酸: {sample['raw_sequence'][:20]}...")
            print("")

        return True

    def get_dataloaders(self, valid_ratio=0.1, test_ratio=0.1):
        """创建训练、验证和测试数据加载器"""
        dataset_size = len(self.dataset)
        print(f"划分数据集 (共{dataset_size}个样本)")
        print(f"- 训练集: {100 * (1 - valid_ratio - test_ratio):.1f}%")
        print(f"- 验证集: {100 * valid_ratio:.1f}%")
        print(f"- 测试集: {100 * test_ratio:.1f}%")

        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        test_split = int(np.floor(test_ratio * dataset_size))
        valid_split = int(np.floor(valid_ratio * dataset_size))

        test_indices = indices[:test_split]
        valid_indices = indices[test_split:test_split + valid_split]
        train_indices = indices[test_split + valid_split:]

        # 创建数据子集
        from torch.utils.data import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # 创建数据加载器
        print("创建数据加载器...")
        train_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=valid_sampler)
        test_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=test_sampler)

        print(f"数据加载器创建完成，训练集: {len(train_indices)}样本")

        return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    import time
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="蛋白质数据处理程序")
    parser.add_argument("--test", action="store_true", help="测试模式，只加载1000个蛋白质")
    parser.add_argument("--max_proteins", type=int, default=None, help="最大蛋白质数量")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--data_dir", type=str, default="./data/protein_data",
                        help="蛋白质数据目录")
    args = parser.parse_args()

    print("=" * 50)
    print("蛋白质数据处理程序")
    print("=" * 50)
    if args.test:
        print("🧪 测试模式: 仅加载有限数量的蛋白质")
    if args.max_proteins:
        print(f"⚠️ 限制最大蛋白质数量: {args.max_proteins}")

    start_time = time.time()

    try:
        # 创建数据处理器
        handler = ProteinDataHandler(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            test_mode=args.test,
            max_proteins=args.max_proteins
        )

        # 测试数据加载
        handler.test_loading(num_samples=3)

        # 获取数据加载器
        train_loader, valid_loader, test_loader = handler.get_dataloaders()

        # 测试加载一个批次
        print("\n测试加载一个批次...")
        batch = next(iter(train_loader))
        print(f"批次大小: {len(batch['protein_id'])}")
        print(f"样本0的蛋白质ID: {batch['protein_id'][0]}")
        print(f"样本0的序列长度: {len(batch['sequence'][0])}")

        total_time = time.time() - start_time
        print(f"\n程序执行完成，总耗时: {total_time:.2f}秒")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()