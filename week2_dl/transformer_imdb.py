"""
Day 11: Transformer 文本分类 (IMDB) —— 本地文件版，无需网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
from pathlib import Path
from tqdm import tqdm
from transformer import TransformerBlock

# ========== 超参数 ==========
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
MAX_LEN = 200
VOCAB_SIZE = 10000
EMBED_DIM = 256
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 本地 IMDB 读取 ==========
class IMDBLocalDataset(Dataset):
    """
    直接读取 aclImdb 目录结构。
    路径: ./data/aclImdb/train/pos/, ./data/aclImdb/train/neg/ 等
    """
    def __init__(self, data_dir: str, split: str):
        self.data = []
        split_dir = Path(data_dir) / split

        # 0 = neg, 1 = pos
        for label, sentiment in enumerate(['neg', 'pos']):
            sentiment_dir = split_dir / sentiment
            if not sentiment_dir.exists():
                raise FileNotFoundError(f"找不到目录: {sentiment_dir}，请先下载并解压 IMDB 数据集")

            for txt_file in sentiment_dir.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                self.data.append((text, label))

        print(f"[{split}] 加载了 {len(self.data)} 条评论")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 使用相对路径，Windows / WSL 均可运行
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'aclImdb')

# 检查数据是否存在
if not Path(DATA_DIR).exists():
    raise FileNotFoundError(
        f"找不到 {DATA_DIR}！\n"
        f"请在 Windows 下载 https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\n"
        f"解压到 D:\\ai_study_2026\\data\\aclImdb，WSL 会自动同步"
    )

train_full = IMDBLocalDataset(DATA_DIR, 'train')
test_dataset = IMDBLocalDataset(DATA_DIR, 'test')

# 划分 train/val (9:1)
train_size = int(0.9 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# ========== 2. 构建词汇表 ==========
def tokenize(text: str) -> list:
    return text.lower().split()

word_counter = Counter()
for text, _ in train_dataset:
    word_counter.update(tokenize(text))

most_common = word_counter.most_common(VOCAB_SIZE - 2)
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in most_common:
    vocab[word] = len(vocab)

print(f"词汇表大小: {len(vocab)}")

# ========== 3. 编码 & DataLoader ==========
def encode(text: str, vocab: dict, max_len: int) -> list:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def collate_fn(batch):
    texts, labels = zip(*batch)
    X = torch.tensor([encode(t, vocab, MAX_LEN) for t in texts], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== 4. 模型定义 ==========
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, num_layers=2, max_len=200, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码 (可学习的 Embedding)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # 堆叠 Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq) — 原始整数 token ID

        # ✅ 在 embedding 之前，用原始 token ID 生成 padding mask
        # PAD token id = 0，所以 x != 0 的位置是有效 token
        mask = (x != 0).float()  # (batch, seq)

        seq_len = x.size(1)

        # 词嵌入 + 位置嵌入
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))

        for block in self.blocks:
            x = block(x, mask)
        x = self.dropout(x)
        # 全局平均池化: (batch, seq, embed) -> (batch, embed)
        x = x.mean(dim=1)

        return self.fc(x)


# 传入所有超参数
model = Transformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=8,
    ff_dim=512,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN,
    num_classes=2
).to(DEVICE)

print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

# ========== 5. 训练 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def run_epoch(model, loader, criterion, optimizer, device, phase='train'):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0, 0, 0

    for data, target in tqdm(loader, desc=phase, leave=False):
        data, target = data.to(device), target.to(device)

        if is_train:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total

print("开始训练...")
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
    val_loss, val_acc = run_epoch(model, val_loader, criterion, None, DEVICE, 'val')

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    print("-" * 40)

test_loss, test_acc = run_epoch(model, test_loader, criterion, None, DEVICE, 'test')
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


#   Train Loss: 0.2395 | Train Acc: 0.9023
#   Val   Loss: 0.4178 | Val   Acc: 0.8324
# ----------------------------------------
#   Test Loss: 0.4481 | Test Acc: 0.8224   
