"""从正式数据集截取少量行，输出到 gleamlm-lite/data/splits/（项目内测试数据）"""
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# 正式数据源
SRC_DIR = os.path.join(_PROJECT_ROOT, 'data', 'lite_data')
# 小测试输出（项目本地，不污染共享 data/）
DST_DIR = os.path.join(_SCRIPT_DIR, 'data', 'splits')

os.makedirs(DST_DIR, exist_ok=True)

n_train = int(os.environ.get('N_TRAIN', 5000))
n_valid = int(os.environ.get('N_VALID', 500))

for split, num, src_name in [("train", n_train, "train.txt"),
                              ("valid", n_valid, "valid.txt")]:
    lines = []
    src = os.path.join(SRC_DIR, src_name)
    if os.path.exists(src):
        with open(src, 'r', encoding='utf-8') as f:
            for i, l in enumerate(f):
                if i >= num:
                    break
                lines.append(l)
    dst = os.path.join(DST_DIR, src_name)
    with open(dst, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'{split}: {len(lines)} 行 -> {dst}')

print(f'\n训练时: --data_dir gleamlm-lite/data/splits')  # 小测试数据放项目本地
