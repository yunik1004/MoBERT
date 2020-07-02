from torch.utils.data import Dataset, IterableDataset
from konlpy.tag import Mecab
from tqdm import tqdm, trange

from transformers import BertTokenizerFast
from itertools import chain
import random
import torch
import json, pickle
import os
from jamo import h2j, j2hcj

from torch.nn.utils.rnn import pad_sequence

class KorquadDataset(Dataset):
    def __init__(self, train=True):
        if train:
            path = "/data/KorQuAD_v1.0_train.json"
            db_name = 'korquad_train.qas'
        else:
            path = "/data/KorQuAD_v1.0_dev.json"
            db_name = 'korquad_dev.qas'
        self.tokenizer = BertTokenizerFast("wiki-vocab.txt")
        
        data = json.load(open(path, encoding='utf-8'))['data']
        
        self.qas = []
        if not os.path.exists(db_name):
            with open(db_name, 'wb') as f:
                self.mecab = Mecab()
                ignored_cnt = 0
                for paragraphs in tqdm(data):
                    paragraphs = paragraphs['paragraphs']
                    for paragraph in paragraphs:
                        _context = paragraph['context']
                        for qa in paragraph['qas']:
                            question = qa['question']
                            answer = qa['answers'][0]['text']
                            input_ids, token_type_ids, start_token_pos, end_token_pos = self.extract_features(_context, question, answer, qa['answers'][0]['answer_start'])
                            if len(input_ids) > 512:
                                if not train:
                                    pickle.dump((input_ids[:510], token_type_ids[:510], -1, -1), f)
                            else:
                                if train:
                                    pickle.dump((input_ids, token_type_ids, start_token_pos, end_token_pos), f)
                                else:
                                    pickle.dump((input_ids, token_type_ids, -1, -1), f)
        
        with open(db_name, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    self.qas.append(data)
                except EOFError:
                    break
            print(len(self.qas))
            
    @property
    def token_num(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.qas)

    def encode(self, line):
        converted_results = map(
            lambda x: x[1:-1],
            self.tokenizer.batch_encode_plus(line)["input_ids"])
        return [2, *chain.from_iterable(converted_results), 3]

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def __getitem__(self, idx):
        return self.qas[idx]
    
    def extract_features(self, context, question, answer, start_char_pos):
        if answer is None:
            # use encode_plus function in tokenizer
            tokenized_q = self.tokenize(question)
            tokenized_c = self.tokenize(context)
            input_ids = [*tokenized_q, *tokenized_c[1:]]
            token_type_ids = [*[0 for _ in tokenized_q], *[1 for _ in tokenized_c[1:]]]
            start_token_pos: int = None
            end_token_pos: int = None
        else:
            # Split sentences using len(answer) and start_char_pos
            context_front = context[:start_char_pos]
            context_back = context[start_char_pos+len(answer):]
            q_ids = self.tokenize(question)
            f_ids = self.tokenize(context_front)
            a_ids = self.tokenize(answer)
            b_ids = self.tokenize(context_back)

            # For processing subwords
            if context_front != '' and context_front[-1] != ' ':
                a_ids = [a_ids[0], a_ids[1], *a_ids[2:]]
            if context_back != '' and context_back[0] != ' ':
                b_ids = [b_ids[0], b_ids[1], *b_ids[2:]]

            # Manually generate input_ids, token_type_ids and start/end_token_pos (carefully remove [CLS] and [SEP])
            input_ids = [*q_ids, *f_ids[1:-1], *a_ids[1:-1], *b_ids[1:]]
            token_type_ids = [*[0 for _ in q_ids], *[1 for _ in f_ids[1:-1]], *[1 for _ in a_ids[1:-1]], *[1 for _ in b_ids[1:]]]
            start_token_pos = len(q_ids) + (len(f_ids) - 2)
            end_token_pos = len(q_ids) + (len(f_ids) - 2) + (len(a_ids) - 2) - 1

        return input_ids, token_type_ids, start_token_pos, end_token_pos
    
    def tokenize(self, sentence):
        if len(sentence) == 0: return [2, 3]
        return self.encode([j2hcj(h2j(word)) for word in self.mecab.morphs(sentence)])
    
    def collate_fn(self, samples):
        input_ids, token_type_ids, start_pos, end_pos = zip(*samples)
        attention_mask = [[1] * len(input_id) for input_id in input_ids]

        input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long) for input_id in input_ids], \
                                 padding_value=0, batch_first=True)
        token_type_ids = pad_sequence([torch.Tensor(token_type_id).to(torch.long) for token_type_id in token_type_ids], \
                                      padding_value=1, batch_first=True)
        attention_mask = pad_sequence([torch.Tensor(mask).to(torch.long) for mask in attention_mask], \
                                      padding_value=0, batch_first=True)

        start_pos = torch.Tensor(start_pos).to(torch.long)
        end_pos = torch.Tensor(end_pos).to(torch.long)

        return input_ids, attention_mask, token_type_ids, start_pos, end_pos

if __name__ == "__main__":
    dataset = KorquadDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=val_dataset.dataset.collate_fn) # shuffle is not needed for validation
    print(next(iter(train_loader)))
    print(next(iter(val_loader)))
    _dataset = KorquadDataset(train=False)
    test_loader = torch.utils.data.DataLoader(_dataset, batch_size=8, shuffle=True, collate_fn=_dataset.collate_fn)
    print(next(iter(test_loader)))

