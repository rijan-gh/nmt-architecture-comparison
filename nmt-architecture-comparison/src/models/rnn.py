import torch
import torch.nn as nn
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)          # hidden: [1, batch_size, hidden_dim]
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_step, hidden):
        # input_step: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_step))  # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)          # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))          # prediction: [batch_size, output_dim]
        return prediction, hidden

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len], trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Pass input sequence through Encoder
        hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_step = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_step, hidden)
            outputs[:, t, :] = output
            
            # Teacher Forcing: use actual target token as next input with probability teacher_forcing_ratio
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_step = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(self, src_tensor, sos_idx, eos_idx, max_len=50):
        """Greedy-decode a single numericalized source sentence.

        Args:
            src_tensor: LongTensor [1, src_len] already on self.device
        Returns:
            token_ids: list[int] of predicted target token ids (no <sos>)
            attentions: None (this architecture has no attention weights)
        """
        self.eval()
        hidden = self.encoder(src_tensor)

        input_step = torch.LongTensor([[sos_idx]]).to(self.device)
        token_ids = []

        for _ in range(max_len):
            output, hidden = self.decoder(input_step, hidden)
            pred_token = output.argmax(1).item()
            if pred_token == eos_idx:
                break
            token_ids.append(pred_token)
            input_step = torch.LongTensor([[pred_token]]).to(self.device)

        return token_ids, None