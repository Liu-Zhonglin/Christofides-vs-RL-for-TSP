import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """The attention mechanism used by the Actor."""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, mask):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        scores = self.V(torch.tanh(self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden_expanded)))
        scores = scores.squeeze(2)
        if mask is not None:
            scores.masked_fill_(mask, -float('inf'))
        return scores


class Actor(nn.Module):
    """
    The Actor network (our Pointer Network). It decides which city to visit next.
    This is the same architecture we finalized before.
    """

    def __init__(self, input_dim, hidden_dim):
        super(Actor, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.decoder_start_input = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device

        encoder_outputs, (hidden, cell) = self.encoder(x)

        tour_indices = []
        tour_log_probs = []

        decoder_input = self.decoder_start_input.repeat(batch_size, 1, 1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # The decoder's hidden state is used as the query for attention
            decoder_hidden_state = hidden.squeeze(0)

            scores = self.attention(encoder_outputs, decoder_hidden_state, mask)
            log_probs = F.log_softmax(scores, dim=1)

            _, selected_index = torch.max(log_probs, dim=1)
            tour_indices.append(selected_index.unsqueeze(1))
            tour_log_probs.append(torch.gather(log_probs, 1, selected_index.unsqueeze(1)))

            mask = mask.clone().scatter_(1, selected_index.unsqueeze(1), True)

            batch_indices = torch.arange(batch_size, device=device)
            decoder_input = x[batch_indices, selected_index, :].unsqueeze(1)

        tour = torch.cat(tour_indices, dim=1)
        log_probability = torch.cat(tour_log_probs, dim=1).sum(dim=1)

        return tour, log_probability, decoder_hidden_state  # Return hidden state for the Critic


class Critic(nn.Module):
    """
    The Critic network. It evaluates the "value" of a state.
    It's a simple MLP that takes the decoder's hidden state as input.
    """

    def __init__(self, hidden_dim):
        super(Critic, self).__init__()
        # A simple 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single value
        )

    def forward(self, state):
        # The 'state' is the hidden state from the Actor's decoder
        value = self.mlp(state)
        return value