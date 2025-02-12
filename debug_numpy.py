import numpy as np

# 替换成你实际的 npz 文件路径
# file_path = '/home/wjx/Xmg_gate/Xmg_gate/examples/data/train/graphs.npz'

file_path = '/home/wjx/npz/final_data/aig_npz/graphs.npz'


try:
    data = np.load(file_path)
    print("加载的数据键：", data.keys())
    x = data['x']
    edge_index = data['edge_index']
    print(f"x 的类型: {type(x)}, shape: {x.shape}")
    print(f"edge_index 的类型: {type(edge_index)}, shape: {edge_index.shape}")
except Exception as e:
    print(f"加载文件时出错：{e}")