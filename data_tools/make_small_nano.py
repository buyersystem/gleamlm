import os, shutil

src = 'data/nano_data/train.txt'
dst_dir = 'data/nano_data'
train = os.path.join(dst_dir, 'train.txt')
backup = os.path.join(dst_dir, 'train_full.txt')

if os.path.getsize(train) > 5000000 and not os.path.exists(backup):
    shutil.copy(train, backup)
    print(f'Backed up train.txt ({os.path.getsize(backup)//1024//1024}MB)')

for split, n in [('train', 2000), ('valid', 200)]:
    out = os.path.join(dst_dir, f'{split}.txt')
    sf = src if split == 'train' else src.replace('train', 'valid')
    with open(sf, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(n)]
    with open(out, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    # Remove old .npy cache
    npy = os.path.join(dst_dir, f'{split}_ids.npy')
    if os.path.exists(npy): os.remove(npy)
    print(f'{split}: {len(lines)} lines, {os.path.getsize(out)//1024}KB')
