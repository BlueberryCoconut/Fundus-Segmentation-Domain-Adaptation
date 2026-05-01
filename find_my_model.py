import os
import torch
import glob
from datetime import datetime

# 你的项目根目录
ROOT_DIR = r'C:\Users\18268\Desktop\PLPB-main'


def scan_models():
    # 递归搜索所有 .pth 结尾的文件
    pth_files = glob.glob(os.path.join(ROOT_DIR, '**', '*.pth'), recursive=True)

    print(f"🔍 雷达扫描完毕，共发现 {len(pth_files)} 个模型文件！")
    print("=" * 60)

    for pth in pth_files:
        try:
            # 获取文件最后修改时间
            mtime = os.path.getmtime(pth)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

            # 读取模型字典
            ckpt = torch.load(pth, map_location='cpu', weights_only=False)

            print(f"📄 文件: {os.path.basename(pth)}")
            print(f"   📂 路径: {pth}")
            print(f"   🕒 最后修改: {mtime_str}")

            if isinstance(ckpt, dict):
                if 'best_dice' in ckpt:
                    # 🌟 重点关注这里！
                    dice = ckpt['best_dice']
                    if dice > 0.8:
                        print(f"   🏆 [天选之子] 记录的最高分数: {dice:.4f} 🔥🔥🔥")
                    else:
                        print(f"   📊 记录的最高分数: {dice:.4f}")

                if 'epoch' in ckpt:
                    print(f"   🔄 保存时的 Epoch: {ckpt['epoch']}")

            print("-" * 60)

        except Exception as e:
            print(f"⚠️ 无法读取 {os.path.basename(pth)}: 可能是纯权重文件或损坏")
            print("-" * 60)


if __name__ == '__main__':
    scan_models()