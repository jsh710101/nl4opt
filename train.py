import argparse
from tqdm import tqdm
from time import perf_counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from transformers import BartTokenizerFast
from data import get_dataset, labels_to_canonical
from model import MyBartForConditionalGeneration
from utils import seed_everything, get_elapsed_time
from scoring import overall_score

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--max_norm', default=1.0, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--disable_tqdm', action='store_true')
args = parser.parse_args()

seed_everything(77)
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')


def collate_fn(batch):
    padding_values = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0, 'labels': -100}

    data = {}
    for key, padding_value in padding_values.items():
        data[key] = [torch.tensor(b[key]) for b in batch]
        data[key] = pad_sequence(data[key], batch_first=True, padding_value=padding_value)

    return data


dataset = {split: get_dataset(split, tokenizer) for split in ['train', 'dev']}
dataloader = {
    split: DataLoader(
        dataset[split], batch_size=args.batch_size, shuffle=(split == 'train'),
        collate_fn=collate_fn, pin_memory=True) for split in ['train', 'dev']}  # num_workers=4

model = MyBartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, args.num_epochs * len(dataloader['train']))

best_accuracy = 0
for epoch in range(1, args.num_epochs + 1):
    start_time = perf_counter()

    model.train()
    train_loss = 0

    for batch in tqdm(dataloader['train'], disable=args.disable_tqdm):
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch).loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * len(batch['labels']) / len(dataset['train'])

    accuracy = 0
    if epoch % args.eval_interval == 0:
        model.eval()

        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader['dev'], disable=args.disable_tqdm):
                batch = {key: value.to(device) for key, value in batch.items()}
                labels += tokenizer.batch_decode(batch.pop('labels').clamp(min=0), skip_special_tokens=True)

                outputs = model.generate(**batch, max_length=200, early_stopping=False, num_beams=4, no_repeat_ngram_size=0)
                preds += tokenizer.batch_decode(outputs, skip_special_tokens=True)

        preds_obj, preds_const = map(list, zip(*[labels_to_canonical(pred) for pred in preds]))
        labels_obj, labels_const = map(list, zip(*[labels_to_canonical(label) for label in labels]))
        accuracy = overall_score(preds_obj, preds_const, labels_obj, labels_const)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            model.save_pretrained('model')

    end_time = perf_counter()
    elapsed_time = get_elapsed_time(start_time, end_time)

    print(f'[Epoch {epoch:3}/{args.num_epochs}] Loss: {train_loss:6.4f} | Accuracy: {accuracy:6.4f} | {elapsed_time}')
print(f'Best Accuracy: {best_accuracy:6.4f}')
