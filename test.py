import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizerFast
from data import get_dataset, labels_to_canonical
from model import MyBartForConditionalGeneration
from utils import seed_everything
from scoring import overall_score

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)
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


dataset = get_dataset('test', tokenizer)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True)  # num_workers=4

model = MyBartForConditionalGeneration.from_pretrained('model').to(device)
model.eval()

preds, labels = [], []
with torch.no_grad():
    for batch in tqdm(dataloader, disable=args.disable_tqdm):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels += tokenizer.batch_decode(batch.pop('labels').clamp(min=0), skip_special_tokens=True)

        outputs = model.generate(**batch, max_length=200, early_stopping=False, num_beams=4, no_repeat_ngram_size=0)
        preds += tokenizer.batch_decode(outputs, skip_special_tokens=True)

preds_obj, preds_const = map(list, zip(*[labels_to_canonical(pred) for pred in preds]))
labels_obj, labels_const = map(list, zip(*[labels_to_canonical(label) for label in labels]))
accuracy = overall_score(preds_obj, preds_const, labels_obj, labels_const)

print(f'Accuracy: {accuracy:6.4f}')
with open('results.out', 'w') as file:
    file.write(str(accuracy))
