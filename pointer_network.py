import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, mask):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        scores = self.V(torch.tanh(self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden_expanded)))
        scores = scores.squeeze(2)
        scores.masked_fill_(mask, -float('inf'))
        return scores


class PointerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PointerNetwork, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.decoder_start_input = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # --- NEW: Get the device from the input tensor ---
        device = x.device

        encoder_outputs, (hidden, cell) = self.encoder(x)

        tour_indices = []
        tour_log_probs = []

        decoder_input = self.decoder_start_input.repeat(batch_size, 1, 1)

        # --- FIX 1: Create the mask on the correct device ---
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # The clone() is no longer needed with the below fix, but is harmless.
            scores = self.attention(encoder_outputs, hidden.squeeze(0), mask)

            log_probs = F.log_softmax(scores, dim=1)

            _, selected_index = torch.max(log_probs, dim=1)
            tour_indices.append(selected_index.unsqueeze(1))
            tour_log_probs.append(torch.gather(log_probs, 1, selected_index.unsqueeze(1)))

            # Update the mask using a non-inplace operation to be safe
            mask = mask.clone().scatter_(1, selected_index.unsqueeze(1), True)

            # --- FIX 2: Create the batch indices on the correct device ---
            batch_indices = torch.arange(batch_size, device=device)
            decoder_input = x[batch_indices, selected_index, :].unsqueeze(1)

        tour = torch.cat(tour_indices, dim=1)
        log_probability = torch.cat(tour_log_probs, dim=1).sum(dim=1)

        return tour, log_probability