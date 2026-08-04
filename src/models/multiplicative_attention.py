import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderWithMultiplicativeAttention(nn.Module):
    def __init__(
        self, 
        input_vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(EncoderWithMultiplicativeAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True  # Use bidirectional for better representations
        )
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to combine bidirectional outputs
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

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
        
        # Combine bidirectional outputs
        outputs = self.fc(outputs)
        
        # Combine bidirectional hidden states
        hidden = self.fc(hidden.transpose(0, 1).reshape(hidden.shape[1], -1)).unsqueeze(0)
        
        return outputs, hidden


class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(MultiplicativeAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Linear layer for multiplicative attention (Luong attention - general)
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, 
        decoder_hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_seq_len, hidden_dim)
            src_mask: (batch_size, src_seq_len) - mask for padding tokens
        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_seq_len)
        """
        batch_size = encoder_outputs.size(0)
        src_seq_len = encoder_outputs.size(1)
        
        # Transform decoder hidden state
        decoder_hidden_transformed = self.attn(decoder_hidden)
        
        # Compute attention scores via dot product (multiplicative)
        # decoder_hidden_transformed: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_seq_len, hidden_dim)
        attention_scores = torch.bmm(
            encoder_outputs, 
            decoder_hidden_transformed.unsqueeze(2)
        ).squeeze(2)
        
        # Apply mask if provided
        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(src_mask == 0, -1e10)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class DecoderWithMultiplicativeAttention(nn.Module):
    def __init__(
        self, 
        output_vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(DecoderWithMultiplicativeAttention, self).__init__()
        
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.attention = MultiplicativeAttention(hidden_dim)
        
        # GRU takes concatenated embedding and context
        self.rnn = nn.GRU(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer takes concatenated context and rnn output
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        input_token: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (num_layers, batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_seq_len, hidden_dim)
            src_mask: (batch_size, src_seq_len)
        Returns:
            prediction: (batch_size, output_vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
            attention_weights: (batch_size, src_seq_len)
        """
        embedded = self.dropout(self.embedding(input_token))
        
        # Get context vector and attention weights
        context, attention_weights = self.attention(hidden.squeeze(0), encoder_outputs, src_mask)
        
        # Concatenate embedding and context
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Concatenate context, rnn output, and embedding for prediction
        output_combined = torch.cat((output.squeeze(1), context, embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(output_combined)
        
        return prediction, hidden, attention_weights


class EncoderDecoderMultiplicativeAttention(nn.Module):
    def __init__(
        self, 
        encoder: EncoderWithMultiplicativeAttention, 
        decoder: DecoderWithMultiplicativeAttention, 
        device: torch.device
    ):
        super(EncoderDecoderMultiplicativeAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_src_mask(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens (0 = PAD)."""
        return (src_tokens != 0).float()

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
        
        # Create source mask
        src_mask = self.create_src_mask(src_tokens)
        
        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src_tokens)
        
        # First input to decoder is <SOS> token
        input_token = tgt_tokens[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_seq_len):
            # Decode step with attention
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
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
        Greedy decoding for inference with attention visualization support.
        
        Args:
            src_tokens: (batch_size, src_seq_len)
            max_length: maximum generation length
            sos_token: start of sequence token index
            eos_token: end of sequence token index
        Returns:
            output_tokens: (batch_size, max_length)
        """
        batch_size = src_tokens.size(0)
        
        # Create source mask
        src_mask = self.create_src_mask(src_tokens)
        
        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src_tokens)
        
        # First input to decoder is <SOS> token
        input_token = torch.LongTensor([sos_token] * batch_size).unsqueeze(1).to(self.device)
        
        output_tokens = torch.zeros(batch_size, max_length).long().to(self.device)
        output_tokens[:, 0] = sos_token
        
        for t in range(1, max_length):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, src_mask)
            top1 = prediction.argmax(1)
            output_tokens[:, t] = top1
            input_token = top1.unsqueeze(1)
            
            # Stop if all sequences have generated EOS
            if (top1 == eos_token).all():
                break
        
        return output_tokens
