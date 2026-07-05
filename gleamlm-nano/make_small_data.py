"""创建小数据集用于快速测试训练流程（输出到独立测试目录，不污染生产数据）"""
import os, shutil

N_TRAIN = 5000
N_VALID = 1000
SRC_BASE = "data/nano_data"
DST_BASE = "data/nano_test_splits"

os.makedirs(DST_BASE, exist_ok=True)

for split, n in [("train", N_TRAIN), ("valid", N_VALID)]:
    src = os.path.join(SRC_BASE, f"{split}.txt")
    dst = os.path.join(DST_BASE, f"{split}.txt")

    if not os.path.exists(src):
        print(f"跳过: {src} 不存在")
        continue

    lines = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line)

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # 删除旧预分词缓存
    npy = os.path.join(DST_BASE, f"{split}_ids.npy")
    if os.path.exists(npy):
        os.remove(npy)

    print(f"{split}: {len(lines)} 行, {os.path.getsize(dst)/1024:.0f} KB -> {dst}")

print(f"\n训练时: --data_dir {DST_BASE}")
