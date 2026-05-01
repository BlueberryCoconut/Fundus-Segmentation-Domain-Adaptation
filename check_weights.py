import torch
from networks.deeplabv3 import DeepLab

# 1. 你的真实模型路径
MODEL_PATH = r'C:\Users\18268\Desktop\PLPB-main\logs\uda_run\best_uda_model.pth'

# 2. 实例化你现在用的模型结构
model = DeepLab(num_classes=3, backbone='mobilenet')

# 3. 加载权重字典
print(f"正在读取文件: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

# 兼容 DataParallel 的 module. 前缀
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 4. 🌟 照妖镜：捕获没对上的参数
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

print("\n" + "="*40)
print(f"❌ 模型需要的，但权重文件里没有的层数 (Missing): {len(missing_keys)}")
if len(missing_keys) > 0:
    print("   比如前 5 个没加载上的层:")
    for k in missing_keys[:5]:
        print(f"    - {k}")

print(f"\n⚠️ 权重文件里有，但模型不需要的层数 (Unexpected): {len(unexpected_keys)}")
if len(unexpected_keys) > 0:
    print("   比如前 5 个多出来的层:")
    for k in unexpected_keys[:5]:
        print(f"    - {k}")
print("="*40)

if len(missing_keys) == 0 and len(unexpected_keys) == 0:
    print("\n🎉 完美匹配！这说明权重 100% 注入了模型！")
else:
    print("\n🚨 警报：模型结构对不上！你跑的一直是个随机初始化的空壳！")