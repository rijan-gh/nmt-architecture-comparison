# Everything related to turning raw text into tensors: vocab, cleaning, and the Dataset class.
import re
import unicodedata
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"
        self.UNK_TOKEN = "<unk>"
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]

        self.word2idx = {}
        self.idx2word = {}

    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"([.!?,¿¡])", r" \1 ", text)
        text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9.!?,¿¡]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return self.normalize_text(text).split()

    def build_vocabulary(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenize(sentence))

        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        idx = len(self.special_tokens)
        for word, count in counter.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        ids = [self.word2idx[self.SOS_TOKEN]]
        ids += [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens]
        ids.append(self.word2idx[self.EOS_TOKEN])
        return ids

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, dataframe, source_vocab, target_vocab):
        self.dataframe = dataframe.reset_index(drop=True)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        source_ids = self.source_vocab.numericalize(row["english"])
        target_ids = self.target_vocab.numericalize(row["spanish"])
        return torch.tensor(source_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def create_collate_fn(source_pad_idx, target_pad_idx):
    def collate_fn(batch):
        sources = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        sources = pad_sequence(sources, batch_first=True, padding_value=source_pad_idx)
        targets = pad_sequence(targets, batch_first=True, padding_value=target_pad_idx)
        return sources, targets
    return collate_fn
