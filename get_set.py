import os
import torch
import deepgate
import torch.nn.functional as F

def process_log_file(log_file, hf, output_dir):
    # 获取文件名（不带扩展名）
    base_name = os.path.basename(log_file).split('.')[0]
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(log_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            a, b = int(parts[0]), int(parts[1])
            label = parts[2]
            
            # 获取 hf[a] 和 hf[b]
            tensor1 = hf[a]
            tensor2 = hf[b]
            cosine_similarity = F.cosine_similarity(tensor1, tensor2, dim=0)
            print(f"Cosine Similarity between node {a} and node {b}: {cosine_similarity.item()}")
            # 沿行拼接
            concatenated_tensor = torch.cat((tensor1, tensor2), dim=0).unsqueeze(0)
            reverse_concatenated_tensor = torch.cat((tensor2, tensor1), dim=0).unsqueeze(0)
            #print(concatenated_tensor.shape)
            # 将拼接后的张量转换为字符串
            concatenated_str = '[' + ', '.join(map(str, concatenated_tensor.squeeze(0).tolist())) + ']'
            reverse_concatenated_str = '[' + ', '.join(map(str, reverse_concatenated_tensor.squeeze(0).tolist())) + ']'

            
            # 根据标签编码
            if label == "EPS":
                label_str = "001"
            elif label == "Kissat":
                label_str = "010"
            elif label == "BDD":
                label_str = "100"
            else:
                continue
            
            # 输出到文件
            out_f.write(f"{concatenated_str} {label_str}\n")
            #out_f.write(f"{reverse_concatenated_str} {label_str}\n")

def main():
    # 输入目录和输出目录
    input_dir = "/home/jwt/uko/pair_log_processed_big"
    output_dir = "/home/jwt/training_set2"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建 DeepGate 模型并加载预训练模型
    model = deepgate.Model()
    model.load_pretrained()
    model.eval()
    
    # 创建 AigParser 并解析 AIG 文件
    parser = deepgate.AigParser(tmp_dir='/home/jwt/uko/tmp')
    
    # 遍历输入目录中的所有 .log 文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".log"):
                log_file = os.path.join(root, file)
                # 获取对应的 AIG 文件路径
                aig_file = os.path.join('/home/jwt/python-deepgate/aigfile/', f"{os.path.splitext(file)[0]}.aiger")
                graph = parser.read_aiger(aig_file)
                hs, hf = model(graph)
                print(f"Processing {log_file}")
                process_log_file(log_file, hf, output_dir)

if __name__ == "__main__":
    main()