import os
import re
import ast
import gc
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = "data/"
LINES_FILE = os.path.join(DATA_PATH, "movie_lines.txt")
CONV_FILE = os.path.join(DATA_PATH, "movie_conversations.txt")

BATCH_SIZE = 8
EPOCHS = 5
MAX_LEN = 64
WINDOW_SIZE = 4
MAX_SAMPLES = 50000

# Критично для памяти!
GRADIENT_ACCUMULATION_STEPS = 2  # Имитирует больший batch size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA (с генераторами)
# =========================
def load_lines(path):
    id2line = {}
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                id2line[parts[0]] = parts[4].strip()
    return id2line

def load_conversations_generator(path, id2line):
    """Генератор вместо загрузки всего в память"""
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                try:
                    line_ids = ast.literal_eval(parts[3])
                    conv = [id2line[i] for i in line_ids if i in id2line]
                    if len(conv) >= 2:
                        yield conv
                except:
                    continue

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# ОПТИМИЗИРОВАННЫЙ DATASET (ленивая токенизация)
# =========================
class LazyGPT2DialogDataset(Dataset):
    """Токенизирует тексты на лету"""
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts  # ← храним только сырые тексты
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Токенизация на лету
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"  # ← сразу тензоры
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0)
        }

# =========================
# EVAL
# =========================
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            with torch.cuda.amp.autocast():  # ← mixed precision для экономии
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                total_loss += outputs.loss.item()
    
    model.train()
    return total_loss / len(loader)

# =========================
# Очистка кэша CUDA
# =========================
def clear_memory():
    """Очистка памяти между эпохами"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# TRAIN
# =========================
def train():
    print("Loading data...")
    id2line = load_lines(LINES_FILE)
    
    print("Building samples...")
    texts = []
    conv_count = 0
    
    # Сбор сэмплов с ограничением
    for conv in load_conversations_generator(CONV_FILE, id2line):
        conv = [clean(x) for x in conv if x]
        
        if len(conv) < 2:
            continue
            
        for i in range(1, len(conv)):
            start = max(0, i - WINDOW_SIZE)
            context = conv[start:i]
            target = conv[i]
            texts.append(" [SEP] ".join(context + [target]))
        
        conv_count += 1
        if MAX_SAMPLES and len(texts) >= MAX_SAMPLES:
            break
    
    print(f"Total samples: {len(texts)}")
    
    # Перемешиваем и берем подмножество
    random.shuffle(texts)
    texts = texts[:MAX_SAMPLES] if MAX_SAMPLES else texts
    
    # =========================
    # SPLIT
    # =========================
    train_texts, temp_texts = train_test_split(texts, test_size=0.2, random_state=42)
    val_texts, test_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)
    
    # Освобождаем память от исходного списка
    del texts, temp_texts
    gc.collect()
    
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    
    # =========================
    # TOKENIZER & MODEL
    # =========================
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Используем ленивый датасет
    train_dataset = LazyGPT2DialogDataset(train_texts, tokenizer, MAX_LEN)
    val_dataset = LazyGPT2DialogDataset(val_texts, tokenizer, MAX_LEN)
    
    # Уменьшаем num_workers для экономии памяти
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=False  # ← workers пересоздаются каждую эпоху
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=False
    )
    
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    
    # Gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()
    
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # =========================
    # EARLY STOPPING
    # =========================
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    
    print("Start training...\n")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_cache=False  # отключаем кэш для экономии памяти
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            progress_bar.set_postfix({
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.3f}",
                "gpu_mem": f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else ""
            })
        
        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained("model/best_model")
            tokenizer.save_pretrained("model/best_model")
            print("✔ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping")
                break
        
        clear_memory()  # очистка памяти после каждой эпохи
    
    print("Training finished")

if __name__ == "__main__":
    train()