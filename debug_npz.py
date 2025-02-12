# import torch
# import numpy as np

# def inspect_npz(npz_path):
#     """
#     简单打印 .npz 文件的结构和内容概要
#     """
#     data = np.load(npz_path, allow_pickle=True)

#     print(f"[INFO] .npz 文件包含以下 keys:")
#     for key in data.keys():
#         print(f"  - {key}")

#     # 如果 'circuits' 是关键字，进一步解析
#     if 'circuits' in data:
#         circuits = data['circuits'].item()  # 假设是一个字典
#         print(f"\n[INFO] 'circuits' 包含 {len(circuits)} 个电路数据")

#         # 遍历每个电路，打印概要信息
#         for idx, (circuit_name, circuit_data) in enumerate(circuits.items()):
#             print(f"  {idx+1}. Circuit Name: {circuit_name}")
#             print(f"     Keys: {list(circuit_data.keys())}")

#             # 对每个 key 打印数据形状或内容类型
#             for key, value in circuit_data.items():
#                 if isinstance(value, np.ndarray):
#                     print(f"        {key}: shape={value.shape}, dtype={value.dtype}")
#                 else:
#                     print(f"        {key}: type={type(value)}")

# # 指定 .npz 文件路径
# # npz_path = "/home/wjx/MixGate1/data/lcm/graphs.npz"
# # npz_path = "/home/wjx/npz/final_data/mig_npz/graphs.npz"
# npz_path = "/home/wjx/npz/final_data/mig_npz/labels.npz"
# # 检查 npz 文件内容
# inspect_npz(npz_path)
import numpy as np

# 指定 npz 文件路径
npz_file_path = "/home/wjx/npz/final_data/newest_npz/mig_npz/labels.npz"

# 加载 npz 文件，允许加载对象数组
npz_data = np.load(npz_file_path, allow_pickle=True)

# 打印 npz 文件中所有的键（变量名）
print("Keys in npz file:", npz_data.files)

# 获取并打印 'labels' 数据
labels = npz_data['labels']
print("Labels content:", labels)

# 遍历并打印每个键对应的数据形状和类型
for key in npz_data.files:
    print(f"Key: {key}, Shape: {npz_data[key].shape}, Type: {npz_data[key].dtype}")

# 关闭文件
npz_data.close()