import os
import re
import random
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import constants directly from config.py
import config

# Special Token Indices
PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
    """Vocabulary tracker for a language."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count PAD, SOS, and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    """Strip accents and convert Unicode to clean ASCII."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """Lowercase, trim spaces, and strip non-letter characters."""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(file_path, reverse=False):
    """Reads dataset file and parses into sentence pairs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at path: {file_path}")

    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang("fra")
        output_lang = Lang("eng")
    else:
        input_lang = Lang("eng")
        output_lang = Lang("fra")

    return input_lang, output_lang, pairs


def filterPair(p):
    """Applies constraints from config.py."""
    return (
        len(p[0].split(' ')) < config.MAX_LENGTH and
        len(p[1].split(' ')) < config.MAX_LENGTH and
        p[1].startswith(config.ENG_PREFIXES)
    )


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(file_path=config.RAW_DATA_PATH, reverse=True):
    """Master preprocessing function: loads, cleans, filters, and builds vocabularies."""
    input_lang, output_lang, pairs = readLangs(file_path, reverse)
    pairs = filterPairs(pairs)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


class TranslationDataset(Dataset):
    """Transforms raw sentence pairs into integer Tensors."""
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.pairs[idx]
        
        src_ids = [self.input_lang.word2index[w] for w in src_sentence.split(' ')] + [EOS_token]
        tgt_ids = [self.output_lang.word2index[w] for w in tgt_sentence.split(' ')] + [EOS_token]
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    """Pads batches dynamically to match length of longest sequence."""
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_token)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_token)
    return src_padded, tgt_padded


def get_dataloaders(file_path=config.RAW_DATA_PATH, batch_size=config.BATCH_SIZE, reverse=True):
    """Creates train and validation PyTorch DataLoaders."""
    input_lang, output_lang, pairs = prepareData(file_path, reverse)
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    
    # 80/20 Train-Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, input_lang, output_lang