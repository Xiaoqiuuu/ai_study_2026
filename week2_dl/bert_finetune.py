"""
Day 12: BERT 微调
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# ============= 配置 =============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
NUM_CLASSES = 2


# 1. 加载预训练 Tokenizer 和 BERT
# bert_size_chinese: 12 层 Transformer, 768 维, 12 头, 约 1 亿参数
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

print(f"BERT 总参数量: {sum(p.numel() for p in bert.parameters()):,}")
# 约 102,267,648

# 2. 构建分类模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, freeze_layers = 10):
        """
        Args:
            freeze_layers: 冻结前 N 层 Transformer, 只训练后面的
        """
        super().__init__()
        self.bert = bert_model

        # 冻结策略: 省显存, 防过拟合, 加速收敛
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for param in self.bert.encoder.layer[:freeze_layers].parameters():
                param.requires_grad = False

            trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.bert.parameters())
            print(f"冻结前 {freeze_layers} 层后, 可训练参数: {trainable:,} / {total:,}")

        # 分类头: 把 [CLS] token 中的 768 维向量映射到类别数
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)        

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len) token ID
        attention_mask: (batch, seq_len) 1 = 有效, 0 = 填充
        """

        # BERT 输出:
        #   last_hidden_state: (batch, seq, 768)    每个token的向量
        #   pooler_output: (bacth, 768)     [CLS] token 经过额外线性层 + tanh
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)

        # 取 [CLS] token 的向量(第 0 个位置)
        # 也可以用 pooler_output, 但 last_hidden_state[:, 0] 更稳定
        cls_vector = outputs.last_hidden_state[:, 0, :] # (batch, 768)

        cls_vector = self.dropout(cls_vector)
        logits = self.classifier(cls_vector)
        return logits

from pathlib import Path
from torch.utils.data import Dataset

class IMDBLocalDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        split_dir = Path(data_dir) / split

        for label, sentiment in enumerate(['neg', 'pos']):
            for txt_file in (split_dir / sentiment).glob('*.txt'):
                with open(txt_file, 'r', encoding = 'utf-8') as f:
                    text = f.read().strip()
                self.data.append((text, label))

        print(f"[{split}] 加载 {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # BERT Tokenizer 自动处理：
        # 1. 分词(WordPiece)
        # 2. 加 [CLS] 和 [SEP]
        # 3. 截断/填充到 max_len
        # 4. 生成 attention_mask
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,      # 加 [CLS] 和 [SEP]
            max_length = self.max_len,
            padding = 'max_length',         # 填充到 max_len
            truncation = True,              # 超长截断
            return_tensors = 'pt'           # 返回 Pytorch 张量
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype = torch.long)
        }
DATA_DIR = Path(__file__).parent.parent / 'data' / 'aclImdb'
train_dataset = IMDBLocalDataset(DATA_DIR, 'train', tokenizer, MAX_LEN)
test_dataset = IMDBLocalDataset(DATA_DIR, 'test', tokenizer, MAX_LEN)

# 划分 val
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, collate_fn = collate_fn)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn = collate_fn)

# 4. 训练

from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


model = BertClassifier(bert, NUM_CLASSES, freeze_layers = 10).to(DEVICE)

# BERT 必须用 AdamW (带权重衰减的 Adam), 不能用普通 Adam
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = LR)

# 学习率预热: 前 10% 步线性升温, 然后线性衰减
total_step = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(0.1 * total_step), num_training_steps = total_step)
criterion = nn.CrossEntropyLoss()

def run_epoch(model, loader, criterion, optimizer, scheduler, device, phase = 'train'):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0, 0, 0
    import tqdm
    for batch in tqdm.tqdm(loader, desc = phase, leave = False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        pred = outputs.argmax(dim =1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

print("开始微调 BERT ...")
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, 'train')
    val_loss, val_acc = run_epoch(model, val_loader, criterion, None, None, DEVICE, 'val')
    print(f"Epoch {epoch + 1} / {EPOCHS}")
    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("-" * 40)


test_loss, test_acc = run_epoch(model, test_loader, criterion, None, None, DEVICE, 'test')
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")