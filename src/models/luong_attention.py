import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # General score alignment option: score(s_t, h_i) = s_t^T * W_a * h_i
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, 1, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        # Transform decoder hidden state: [batch_size, 1, hidden_dim]
        score = self.W_a(decoder_hidden)
        
        # Multiplicative batch matrix multiplication: [batch_size, 1, src_len]
        alignment_scores = torch.bmm(score, encoder_outputs.transpose(1, 2))
        
        # Softmax over source sequence length dimension
        attention_weights = F.softmax(alignment_scores, dim=2)
        return attention_weights


class EncoderLuongLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class DecoderLuongLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention, dropout_p=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Luong passes token embedding directly into LSTM
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        
        # Combine current LSTM state and context vector before final projection
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        # input_step: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_step))  # [batch_size, 1, emb_dim]

        # 1. Generate current decoder hidden state
        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # [batch_size, 1, hidden_dim]

        # 2. Calculate Luong multiplicative attention using CURRENT decoder output
        attn_weights = self.attention(rnn_output, encoder_outputs)  # [batch_size, 1, src_len]

        # 3. Calculate weighted context vector: [batch_size, 1, hidden_dim]
        context = torch.bmm(attn_weights, encoder_outputs)

        # 4. Concatenate context vector and current decoder state
        concat_input = torch.cat((rnn_output, context), dim=2)  # [batch_size, 1, hidden_dim * 2]
        concat_output = torch.tanh(self.concat(concat_input))   # [batch_size, 1, hidden_dim]

        # 5. Output prediction layer
        prediction = self.fc_out(concat_output.squeeze(1))      # [batch_size, output_dim]

        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2SeqLuongAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input_step = trg[:, 0].unsqueeze(1)  # <sos> token

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_step, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_step = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(self, src_tensor, sos_idx, eos_idx, max_len=50):
        """Greedy-decode a single numericalized source sentence.

        Returns:
            token_ids: list[int] of predicted target token ids (no <sos>)
            attentions: FloatTensor [trg_len, src_len] of attention weights
        """
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src_tensor)

        input_step = torch.LongTensor([[sos_idx]]).to(self.device)
        token_ids = []
        attentions = []

        for _ in range(max_len):
            output, hidden, cell, attn_weights = self.decoder(input_step, hidden, cell, encoder_outputs)
            attentions.append(attn_weights.squeeze(0).cpu())
            pred_token = output.argmax(1).item()
            if pred_token == eos_idx:
                break
            token_ids.append(pred_token)
            input_step = torch.LongTensor([[pred_token]]).to(self.device)

        return token_ids, torch.stack(attentions) if attentions else None