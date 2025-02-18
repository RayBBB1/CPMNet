import torch

# 假設你的 .pth 檔案路徑是 'your_model.pth'
pth_file_path = '/root/notebooks/automl/CPMNet/save/[2025-02-17-1050]_ME_db1/model/epoch_220.pth'

try:
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    print(checkpoint)
    print(f"載入的物件類型: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"字典的鍵 (keys): {checkpoint.keys()}")
    elif isinstance(checkpoint, torch.nn.Module):
        print(f"載入的是模型物件: {checkpoint}")
    else:
        print(f"載入的物件: {checkpoint}")

except Exception as e:
    print(f"載入錯誤: {e}")