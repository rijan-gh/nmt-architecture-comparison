import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim] (top layer hidden state)
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times: [batch_size, src_len, hidden_dim]
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate alignment energy: [batch_size, src_len, 1]
        energy = torch.tanh(self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs))
        
        # Compute attention weights: [batch_size, src_len]
        attention_weights = F.softmax(self.v_a(energy).squeeze(2), dim=1)
        return attention_weights


class EncoderAttentionLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden, cell: [num_layers, batch_size, hidden_dim]
        return outputs, hidden, cell


class DecoderAttentionLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention, dropout_p=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # Input to LSTM is concatenated embedding vector + context vector
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        # input_step: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_step))  # [batch_size, 1, emb_dim]

        # Compute attention weights using top layer hidden state
        # hidden[-1]: [batch_size, hidden_dim]
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        attn_weights_expanded = attn_weights.unsqueeze(1)          # [batch_size, 1, src_len]

        # Calculate weighted context vector: [batch_size, 1, hidden_dim]
        context = torch.bmm(attn_weights_expanded, encoder_outputs)

        # Concatenate embedded token and context vector for LSTM input
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hidden_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Prediction combines output, context, and embedding
        prediction_input = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(prediction_input)

        return prediction, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
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