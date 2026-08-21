# Encoder-Decoder architecture (Sutskever et al., 2014): LSTM encoder compresses the
# source sentence into a fixed context vector that initializes the LSTM decoder. No attention.
import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=2, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout_p if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=2, dropout_p=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout_p if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_step, hidden, cell):
        embedded = self.dropout(self.embedding(input_step))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input_step = trg[:, 0].unsqueeze(1)  # <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1).unsqueeze(1)
            input_step = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

        return outputs

    @torch.no_grad()
    def translate(self, src_tensor, sos_idx, eos_idx, max_len=50):
        self.eval()
        hidden, cell = self.encoder(src_tensor)
        input_step = torch.LongTensor([[sos_idx]]).to(self.device)
        token_ids = []

        for _ in range(max_len):
            output, hidden, cell = self.decoder(input_step, hidden, cell)
            pred_token = output.argmax(1).item()
            if pred_token == eos_idx:
                break
            token_ids.append(pred_token)
            input_step = torch.LongTensor([[pred_token]]).to(self.device)

        return token_ids, None
