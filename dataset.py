from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from transformers import BertTokenizerFast
from itertools import chain
import random
import torch

from torch.nn.utils.rnn import pad_sequence


class WikiDataset(Dataset):
    def __init__(self, path=("data/data.txt", "data/pos.txt"), lazy=False):
        self.tokenizer = BertTokenizerFast("wiki-vocab.txt")
        self.paragraphs = [[]]
        self.lazy = lazy
        self.pos_labels = set([])

        valid = True
        with open(path[0], encoding="utf-8") as f_data:
            with open(path[1], encoding="utf-8") as f_pos:
                for d, p in tqdm(zip(f_data, f_pos), desc="load_data"):
                    if len(d.strip()) == 0:
                        """
                        if len(self.paragraphs) > 100:
                            break
                        """
                        if len(self.paragraphs[-1]) > 0:
                            self.paragraphs.append([])
                        else:
                            valid = True
                    elif valid:
                        _d, _p = d.strip().split(), p.strip().split()
                        if len(_p) > 256:
                            valid = False
                            self.paragraphs[-1] = []
                        else:
                            self.paragraphs[-1].append((_d, _p))
                            self.pos_labels |= set(_p)

        self.pos_labels_to_ids = {}
        for i, pos_label in enumerate(sorted(self.pos_labels)):
            self.pos_labels_to_ids[pos_label] = i + 1

    @property
    def token_num(self):
        return self.tokenizer.vocab_size

    @property
    def pos_num(self):
        return len(self.pos_labels)

    def __len__(self):
        return len(self.paragraphs) - 1

    def encode_line(self, d, p):
        converted_results = map(
            lambda x: (x[0][1:-1], [self.pos_labels_to_ids[x[1]]] * (len(x[0]) - 2)),
            zip(self.tokenizer.batch_encode_plus(d)["input_ids"], p),
        )
        token_ids, pos_ids = zip(*converted_results)
        return list(chain.from_iterable(token_ids)), list(chain.from_iterable(pos_ids))

    def __getitem__(self, idx):
        token_ids, pos_ids = zip(
            *list(map(lambda x: self.encode_line(*x), self.paragraphs[idx]))
        )
        return token_ids, pos_ids


class PretrainDataset(IterableDataset):
    def __init__(self, dataset: WikiDataset):
        self.dataset = dataset

    @property
    def token_num(self):
        return self.dataset.token_num

    @property
    def pos_num(self):
        return self.dataset.pos_num

    def __iter__(self):
        CLS = 2
        SEP = 3
        MSK = 4
        TOKEN_NUM = self.token_num

        while True:
            source_sentences: List[int] = None
            MLM_sentences: List[int] = None
            MLM_mask: List[bool] = None
            NSP_label: bool = random.random() < 0.5
            if NSP_label:
                while True:
                    target_idx = random.randrange(
                        len(self.dataset)
                    )  # sample random index
                    first_paragraph, first_pos_ids = self.dataset[target_idx]
                    second_paragraph, second_pos_ids = first_paragraph, first_pos_ids
                    if len(first_paragraph) == 1:
                        continue  # if cannot sample two sentences in the paragraph
                    row_idx = random.randrange(len(first_paragraph) - 1)
                    first_idx, second_idx = row_idx, row_idx + 1
                    break
            else:
                first_idx = random.randrange(
                    len(self.dataset)
                )  # sample two random indices
                second_idx = (
                    first_idx + random.randrange(len(self.dataset) - 1) + 1
                ) % len(
                    self.dataset
                )  # prevent sampling same paragraph
                first_paragraph, first_pos_ids = self.dataset[first_idx]
                second_paragraph, second_pos_ids = self.dataset[second_idx]
                first_idx, second_idx = (
                    random.randrange(len(first_paragraph) - 1),
                    random.randrange(len(second_paragraph) - 1),
                )

            source_sentences = [
                CLS,
                *first_paragraph[first_idx],
                SEP,
                *second_paragraph[second_idx],
                SEP,
            ]  # sample random connected sentences in the selected paragraph
            MT_label = [0, *first_pos_ids[first_idx], 0, *second_pos_ids[second_idx], 0]
            token_type_ids = [
                0,
                *[0 for _ in first_pos_ids[first_idx]],
                0,
                *[1 for _ in second_pos_ids[second_idx]],
                1,
            ]

            MLM_mask = [True for token in source_sentences]
            n_false = int(
                (0.85 * (len(source_sentences) - 3))
            )  # number of tokens without masking
            n_true = (
                len(source_sentences) - 3 - n_false
            )  # number of tokens with masking
            n_mask = int(0.8 * n_true)  # number of <MSK> tokens
            n_same = (n_true - n_mask) // 2  # number of same mask
            n_random = n_true - (n_mask + n_same)  # number of random mask

            MLM_sentences = source_sentences[:]
            cntt = 0  # variable for debugging
            for i, token in enumerate(source_sentences):
                if token == CLS or token == SEP:
                    continue
                choice = random.randrange(
                    n_false + n_mask + n_same + n_random
                )  # use uniformly random
                # update counter for each case
                if choice < n_false:
                    n_false -= 1  # masking will not be applied
                else:
                    MLM_mask[i] = False  # masking will be applied
                    if choice < n_false + n_mask:
                        n_mask -= 1
                        MLM_sentences[i] = MSK  # masking type: <MSK>
                    elif choice < n_false + n_mask + n_same:
                        n_same -= 1  # masking type: same
                    else:
                        n_random -= 1  # masking type: random
                        new_token = (
                            token + 1 + random.randrange(TOKEN_NUM - 1 - 5)
                        ) % TOKEN_NUM  # prevent sampling same token
                        if new_token < token:
                            new_token += 5  # prevent sampling special token
                        MLM_sentences[i] = new_token
                if MLM_sentences[i] == MSK:
                    cntt += 1  # for debugging
            ### END YOUR CODE

            assert len(source_sentences) == len(MLM_sentences) == len(MLM_mask)
            yield source_sentences, MLM_sentences, MLM_mask, NSP_label, MT_label, token_type_ids


def pretrain_collate_fn(samples):
    """ Sentence collate function
    
    How to Use:
    data_loader = DataLoader(sent_dataset, ..., collate_fn=pretrain_collate_fn, ...)
    """
    PAD = 0
    src, mlm, mask, nsp, mt, token_type_ids = zip(*samples)
    src = pad_sequence(
        [torch.Tensor(sentence).to(torch.long) for sentence in src],
        padding_value=PAD,
        batch_first=True,
    )
    mlm = pad_sequence(
        [torch.Tensor(sentence).to(torch.long) for sentence in mlm],
        padding_value=PAD,
        batch_first=True,
    )
    mask = pad_sequence(
        [torch.Tensor(m).to(torch.float) for m in mask],
        padding_value=True,
        batch_first=True,
    )
    nsp = torch.Tensor(nsp).to(torch.long)
    mt = pad_sequence(
        [torch.Tensor(sentence).to(torch.long) for sentence in mt],
        padding_value=0,
        batch_first=True,
    )
    token_type_ids = pad_sequence(
        [torch.Tensor(sentence).to(torch.long) for sentence in token_type_ids],
        padding_value=1,
        batch_first=True,
    )
    return src, mlm, mask, nsp, mt, token_type_ids


if __name__ == "__main__":
    dataset = PretrainDataset(WikiDataset())
    batch = []
    for i, hi in enumerate(iter(dataset)):
        print(hi)
        batch.append(hi)
        if i > 5:
            break
    print(pretrain_collate_fn(batch))
