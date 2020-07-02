from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm, trange

from transformers import BertTokenizerFast
from itertools import chain
import random
import torch

from torch.nn.utils.rnn import pad_sequence


class NaverSentimentDataset(Dataset):
    def __init__(self, train=True):
        if train:
            path = ("/data/data_nsmc_train.txt", "/data/rating_nsmc_train.txt")
        else:
            path = ("/data/data_nsmc_test.txt", "/data/rating_nsmc_test.txt")
        self.tokenizer = BertTokenizerFast("wiki-vocab.txt")
        self.reviews = [[]]
        
        valid = True
        with open(path[0], encoding="utf-8") as f_data:
            with open(path[1], encoding="utf-8") as f_rating:
                self.reviews = [(d.strip().split(), int(r)) for d, r, in tqdm(zip(f_data, f_rating), desc="load_data")]
                
    @property
    def token_num(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.reviews)

    def encode(self, line):
        converted_results = map(
            lambda x: x[1:-1],
            self.tokenizer.batch_encode_plus(line)["input_ids"])
        return [2, *chain.from_iterable(converted_results), 3]

    def __getitem__(self, idx):
        input_ids, rating = self.encode(self.reviews[idx][0]), self.reviews[idx][1]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        return input_ids, token_type_ids, attention_mask, rating

    def collate_fn(self, samples):
        """ Sentence collate function

        How to Use:
        data_loader = DataLoader(sent_dataset, ..., collate_fn=pretrain_collate_fn, ...)
        """
        PAD = 0
        input_ids, token_type_ids, attention_masks, ratings = zip(*samples)
        input_ids = pad_sequence(
            [torch.Tensor(sentence).to(torch.long) for sentence in input_ids],
            padding_value=PAD,
            batch_first=True,
        )
        token_type_ids = pad_sequence(
            [torch.Tensor(sentence).to(torch.long) for sentence in token_type_ids],
            padding_value=1,
            batch_first=True,
        )
        attention_masks = pad_sequence(
            [torch.Tensor(m).to(torch.float) for m in attention_masks],
            padding_value=0,
            batch_first=True,
        )
        ratings = torch.Tensor(ratings).to(torch.long)
        return input_ids, token_type_ids, attention_masks, ratings


if __name__ == "__main__":
    dataset = NaverSentimentDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.dataset.collate_fn) # shuffle is not needed for validation
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))
