import torch

def check_cuda_and_gpu():
    print("=== 检查 CUDA 和 GPU ===")
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("[INFO] CUDA 可用")
        print(f"[INFO] CUDA 版本: {torch.version.cuda}")
        print(f"[INFO] PyTorch CUDA 支持: {torch.backends.cudnn.version()}")
        print(f"[INFO] GPU 设备数量: {torch.cuda.device_count()}")
        
        # 打印所有可用的 GPU 信息
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 检查当前默认 GPU
        print(f"[INFO] 当前设备: {torch.cuda.current_device()}")
        print(f"[INFO] 当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("[ERROR] CUDA 不可用，请检查 CUDA 驱动是否安装正确")

if __name__ == "__main__":
    check_cuda_and_gpu()
