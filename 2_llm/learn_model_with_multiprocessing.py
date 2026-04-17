import os
import re
import ast
import math
import time
import torch
import multiprocessing

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# =========================
# CONFIG
# =========================
DATA_PATH = "data/"
LINES_FILE = os.path.join(DATA_PATH, "movie_lines.txt")
CONV_FILE = os.path.join(DATA_PATH, "movie_conversations.txt")

BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 128
WINDOW_SIZE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD DATA
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
    conversations = []
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                line_ids = ast.literal_eval(parts[3])
                conv = [id2line[i] for i in line_ids if i in id2line]
                conversations.append(conv)
    return conversations


# =========================
# CLEANING
# =========================
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# BUILD WINDOWS (GPT STYLE)
# =========================
def build_windows(conversations, window_size=4):
    samples = []

    for conv in conversations:
        conv = [clean(x) for x in conv if x]

        if len(conv) < 2:
            continue

        for i in range(1, len(conv)):
            start = max(0, i - window_size)
            context = conv[start:i]
            target = conv[i]

            samples.append((context, target))

    return samples


SEP = " [SEP] "

def to_text(context, target):
    return SEP.join(context + [target])


# =========================
# DATASET
# =========================
class GPT2DialogDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }


# =========================
# TRAIN
# =========================
def train():
    print("Loading data...")
    id2line = load_lines(LINES_FILE)
    conversations = load_conversations(CONV_FILE, id2line)

    print("Building samples...")
    samples = build_windows(conversations, WINDOW_SIZE)
    dataset_texts = [to_text(c, t) for c, t in samples]

    print("Total samples:", len(dataset_texts))

    print("Tokenizing...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        dataset_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    dataset = GPT2DialogDataset(encodings)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,          # ← теперь можно
        pin_memory=True
    )

    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()

    print("Start training...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if i % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        ppl = math.exp(avg_loss)

        print(f"\nEpoch {epoch+1} DONE")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Perplexity: {ppl:.2f}")
        print(f"Time: {time.time() - start_time:.2f}s")
        print("-" * 40)

    # =========================
    # SAVE
    # =========================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"model/gpt2-dialog_{timestamp}"

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Model saved to:", save_path)


# =========================
# ENTRY POINT (ВАЖНО!)
# =========================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train()