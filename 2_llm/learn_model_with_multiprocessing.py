import os
import re
import ast
import gc
import math
import time
import random
import torch
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# =========================
# CONFIG
# =========================
DATA_PATH = "data/"
LINES_FILE = os.path.join(DATA_PATH, "movie_lines.txt")
CONV_FILE = os.path.join(DATA_PATH, "movie_conversations.txt")

MODEL_DIR = "model"

BATCH_SIZE = 8
EPOCHS = 2
MAX_LEN = 64
WINDOW_SIZE = 4
MAX_SAMPLES = 50000

GRAD_ACCUM = 2
LR = 3e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA LOAD
# =========================
def load_lines(path):
    id2line = {}
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                id2line[parts[0]] = parts[4].strip()
    return id2line


def load_conversations(path, id2line):
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                try:
                    ids = ast.literal_eval(parts[3])
                    conv = [id2line[i] for i in ids if i in id2line]
                    if len(conv) >= 2:
                        yield conv
                except:
                    continue


# =========================
# CLEAN
# =========================
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# DATASET
# =========================
class LazyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }


# =========================
# EVAL
# =========================
def evaluate(model, loader):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn,
                labels=input_ids
            )
            losses.append(outputs.loss.item())

    model.train()
    return sum(losses) / len(losses)


# =========================
# MEMORY CLEAN
# =========================
def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# TRAIN
# =========================
def train():
    print("Loading data...")

    id2line = load_lines(LINES_FILE)

    texts = []
    for conv in load_conversations(CONV_FILE, id2line):
        conv = [clean(x) for x in conv if x]

        for i in range(1, len(conv)):
            start = max(0, i - WINDOW_SIZE)
            context = conv[start:i]
            target = conv[i]
            texts.append(" [SEP] ".join(context + [target]))

        if len(texts) >= MAX_SAMPLES:
            break

    random.shuffle(texts)
    texts = texts[:MAX_SAMPLES]

    train_texts, temp = train_test_split(texts, test_size=0.2, random_state=42)
    val_texts, _ = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Train={len(train_texts)} Val={len(val_texts)}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = LazyDataset(train_texts, tokenizer, MAX_LEN)
    val_ds = LazyDataset(val_texts, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    # =========================
    # LOGS
    # =========================
    logs = []

    best_val = float("inf")
    patience = 2
    wait = 0

    print("Training started...")

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        optimizer.zero_grad()

        for step, batch in enumerate(bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=input_ids,
                    use_cache=False
                )
                loss = out.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM

            bar.set_postfix(loss=loss.item() * GRAD_ACCUM)

            global_step += 1

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        ppl = math.exp(val_loss)

        lr = scheduler.get_last_lr()[0]

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": ppl,
            "lr": lr
        })

        print(f"\nEpoch {epoch+1}")
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={ppl:.2f}")

        # =========================
        # EARLY STOP
        # =========================
        if val_loss < best_val:
            best_val = val_loss
            wait = 0

            # SAVE ONLY WEIGHTS
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(MODEL_DIR, f"gpt2_weights_{ts}.pt")

            torch.save(model.state_dict(), save_path)
            print("Saved:", save_path)

        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

        clear_mem()

    # =========================
    # SAVE LOGS
    # =========================
    df = pd.DataFrame(logs)
    os.makedirs(MODEL_DIR, exist_ok=True)

    log_path = os.path.join(MODEL_DIR, "train_log.csv")
    df.to_csv(log_path, index=False)

    print("Logs saved:", log_path)


if __name__ == "__main__":
    train()