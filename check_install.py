import torch
import torchvision

print("PyTorch 版本：", torch.__version__)
print("Torchvision 版本：", torchvision.__version__)

cuda_available = torch.cuda.is_available()
mps_available = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)

print("CUDA 可用：", cuda_available)
print("MPS 可用：", mps_available)

if cuda_available:
    device = torch.device("cuda")
    print("NVIDIA 显卡：", torch.cuda.get_device_name(0))
elif mps_available:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("最终使用设备：", device)

x = torch.rand(2, 3, device=device)

print("测试张量：")
print(x)
print("测试张量所在设备：", x.device)