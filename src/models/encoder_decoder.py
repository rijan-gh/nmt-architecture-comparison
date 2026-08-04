import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, 
        input_vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: (batch_size, src_seq_len)
        Returns:
            outputs: (batch_size, src_seq_len, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(src_tokens))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(
        self, 
        output_vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(Decoder, self).__init__()
        
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        input_token: torch.Tensor, 
        hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (num_layers, batch_size, hidden_dim)
        Returns:
            prediction: (batch_size, output_vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class EncoderDecoderSeq2Seq(nn.Module):
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        device: torch.device
    ):
        super(EncoderDecoderSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self, 
        src_tokens: torch.Tensor, 
        tgt_tokens: torch.Tensor, 
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src_tokens: (batch_size, src_seq_len)
            tgt_tokens: (batch_size, tgt_seq_len)
            teacher_forcing_ratio: probability of using ground truth as input
        Returns:
            outputs: (batch_size, tgt_seq_len, output_vocab_size)
        """
        batch_size = src_tokens.size(0)
        tgt_seq_len = tgt_tokens.size(1)
        output_vocab_size = self.decoder.output_vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_seq_len, output_vocab_size).to(self.device)
        
        # Encode source sequence
        _, hidden = self.encoder(src_tokens)
        
        # First input to decoder is <SOS> token
        input_token = tgt_tokens[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_seq_len):
            # Decode step
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = prediction
            
            # Teacher forcing: use ground truth or predicted token
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt_tokens[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs

    def translate(
        self, 
        src_tokens: torch.Tensor, 
        max_length: int = 50,
        sos_token: int = 1,
        eos_token: int = 2
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        
        Args:
            src_tokens: (batch_size, src_seq_len)
            max_length: maximum generation length
            sos_token: start of sequence token index
            eos_token: end of sequence token index
        Returns:
            output_tokens: (batch_size, max_length)
        """
        batch_size = src_tokens.size(0)
        
        # Encode source sequence
        _, hidden = self.encoder(src_tokens)
        
        # First input to decoder is <SOS> token
        input_token = torch.LongTensor([sos_token] * batch_size).unsqueeze(1).to(self.device)
        
        output_tokens = torch.zeros(batch_size, max_length).long().to(self.device)
        output_tokens[:, 0] = sos_token
        
        for t in range(1, max_length):
            prediction, hidden = self.decoder(input_token, hidden)
            top1 = prediction.argmax(1)
            output_tokens[:, t] = top1
            input_token = top1.unsqueeze(1)
            
            # Stop if all sequences have generated EOS
            if (top1 == eos_token).all():
                break
        
        return output_tokens
