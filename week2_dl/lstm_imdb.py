"""
LSTM 文本分类(IMDB)
目标: 理解序列建模、Embedding、Padding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

#=========== 超参数 ==========
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
MAX_LEN = 200           #每条评论最多保留200个词
VOCAB_SIZE = 10000      #只保留频率最高的10000个词，其余标为<UNK>
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========== 数据准备 ==========

dataset = load_dataset("imdb")
train_texts = [item["text"] for item in dataset["train"]]
train_lables = [item["lables"] for item in dataset["train"]]
test_texts = [item["text"] for item in dataset["test"]]
test_lables = [item["lables"] for item in dataset["test"]]

print(f"训练集: {len(train_texts)} 条, 测试集: {len(test_texts)} 条")

#简单分词: 按空格分
def tokenize(text):
    return text.lower().split()

#构建词汇表
from collections import Counter
word_counter = Counter()
for text in train_texts:
    word_counter.update(tokenize(text))

#取最常见的 VOCAL_SIZE -2 个词， 预留 <PAD>=0, <UNK>=1
most_common = word_counter.most_common(VOCAB_SIZE - 2)
vocab = {"<PAD>": 0, "UNK": 1}
for word, _ in most_common:
    vocab[word] = len(vocab)

print(f"词汇表大小: {len(vocab)}")

#文本 -> 索引序列
def encode(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    #截断或填充到max_len
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

#编码全部数据
x_train = torch.tensor([encode(t, vocab, MAX_LEN) for t in train_texts], dtype = torch.long)
y_train = torch.tensor(train_lables, dtype = torch.long)
x_test = torch.tensor([encode(t, vocab, MAX_LEN) for t in test_texts], dtype = torch.long)
y_test = torch.tensor(test_lables, dtype = torch.long)

# DataLoader
train_loader = DataLoader(list(zip(x_train, y_train)), batch_size = BATCH_SIZE, shuffle =True)
test_loader = DataLoader(list(zip(x_test, y_test)), batch_size = BATCH_SIZE, shuffle = False)

#模型定义
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()

        # Embedding: 把词 ID (0~9999) 映射成稠密向量(embed_dim,)
        # 参数量: vocab_size * embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        #LSTM: 处理序列, 输出每个时间步的 hidden state
        #batch_first = True, 表示输入是 (batch, seq, feature) 而不是(seq, batch, feature)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True, dropout = 0.3)
        # Dropout 在多层 LSTM 的层之间生效；

        # 分类头: 用最后一个时间步的 hidden state 做二分类
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len) = (32, 200)

        # 1. embedding: (32, 200) -> (32, 200, 128)
        embedded = self.embedding(x)

        # 2. LSTM:
        # output: (32, 200, 256) 所有时间步的 hidden state
        # (h_n, c_n): h_n 是(2, 32, 256) 最后一层, 最后一步的 hidden
        output, (h_n, c_n) = self.lstm(embedded)

        # 3. 取最后一个时间步的 hidden state
        # h_n 形状: (num_layers, batch, hidden_dim)
        # 我们要最后一层: h_n[-1] -> (32, 256)
        last_hidden = h_n[-1]

        # 4. 分类
        logits = self.fc(last_hidden)
        return logits

model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes= 2).to(DEVICE)

#参数量统计
total = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total:,}")

# =================训练 & 测试 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LR)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim =1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(loader), correct / total

def test_epoch(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model.forward(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim = 1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"Batch: {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(loader), correct / total

# ================ 主循环 =================
print("开始训练 LSTM ...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device = DEVICE)
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device = DEVICE)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"    Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("-" * 40)

print("训练完成！")