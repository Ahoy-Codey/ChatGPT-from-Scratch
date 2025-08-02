
---

# ChatGPT from Scratch

This project implements a minimal GPT-style Transformer language model in PyTorch, using [tiktoken](https://github.com/openai/tiktoken) for fast, GPT-2 compatible tokenization.  
You can train the model from scratch on your own text, and fine-tune it on new data for custom tasks.
Try to run it in [Google Colab](https://colab.research.google.com/) for getting the model to be trained quicker.

---

## Features


- **PyTorch** implementation, CUDA-ready
- **tiktoken** tokenizer (GPT-2 BPE)
- Custom dataset loader for any `.txt` file
- Easy training and fine-tuning
- Simple text generation

---

## Getting Started

### 1. **Install Requirements**

```bash
pip install torch tiktoken
```

---

### 2. **Prepare Your Data**

- Place your main training text in `data.txt`.
- For fine-tuning, place your new data in `finetune.txt`.

---

### 3. **Train the Base GPT Model**

Run the notebook or script to train from scratch:

```python
import tiktoken
from torch.utils.data import DataLoader

# Tokenizer and dataset
tokenizer = tiktoken.get_encoding("gpt2")
dataset = TiktokenDataset("data.txt", tokenizer, block_size=128)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model config and training (see notebook for full code)
model = GPT(config).to(device)
# ... training loop ...
torch.save(model.state_dict(), 'gpt_tiktoken.pth')
```

---

### 4. **Fine-tune the Model**

To fine-tune on new data:

```python
# Load pre-trained model
model.load_state_dict(torch.load('gpt_tiktoken.pth'))

# Prepare fine-tuning dataset
finetune_dataset = TiktokenDataset("finetune.txt", tokenizer, block_size=128)
finetune_loader = DataLoader(finetune_dataset, batch_size=16, shuffle=True)

# Lower learning rate for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop (see notebook for full code)
# ... fine-tuning loop ...
torch.save(model.state_dict(), 'gpt_tiktoken_finetuned.pth')
```

---

### 5. **Generate Text**

Generate text with your trained or fine-tuned model:

```python
def sample(model, tokenizer, start_text, length=100, temperature=1.0):
    model.eval()
    idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(next(model.parameters()).device)
    for _ in range(length):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())

# Example:
print(sample(model, tokenizer, start_text="Once upon a time", length=50))
```

---

## Files

- `gpt_tiktoken.pth` — Base GPT model checkpoint (after training on `data.txt`)
- `gpt_tiktoken_finetuned.pth` — Fine-tuned model checkpoint (after training on `finetune.txt`)
- `data.txt` — Your base training data
- `finetune.txt` — Your fine-tuning data
- `notebook.ipynb` — Jupyter notebook with all code

---
## Customization

- Change model size (`n_layers`, `n_heads`, `n_embd`) in `GPTConfig` for larger or smaller models.
- Use any `.txt` file for training or fine-tuning.
- Adjust `block_size`, `batch_size`, and learning rates as needed.

---