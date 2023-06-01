import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', type=str, help='File path')
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')

    args = parser.parse_args()

    # Load the Stable Diffusion model
    model_name = 'openai/stable-diffusion'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and preprocess the training data
    train_dataset = TextDataset(args.train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Training loop
    model.train()
    model.to('cuda')
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False)
        for batch in progress_bar:
            inputs = batch.to('cuda')

            # Forward pass
            outputs = model(inputs, labels=inputs)

            # Loss calculation
            loss = outputs.loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({'loss': loss.item()})

    # Save the trained model weights
    model.save_pretrained(args.fp)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        self.texts = self.load_data(file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        encoding = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return encoding.input_ids.squeeze(0)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            texts = file.read().splitlines()
        return texts

if __name__ == '__main__':
    main()
