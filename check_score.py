import torch

# 把这里换成你实际的 checkpoint_latest.pth.tar 的路径
ckpt_path = r'C:\Users\18268\Desktop\PLPB-main\logs\Domain1\20260429_224100.174927\checkpoint_latest.pth.tar'

try:
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f"🔥 模型训练到了第 {checkpoint.get('epoch', '未知')} 轮")
    print(f"🏆 验证集历史最高 Dice 分数: {checkpoint.get('best_mean_dice', 0.0):.4f}")
except Exception as e:
    print("读取失败，请检查路径对不对~", e)