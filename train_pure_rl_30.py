import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# We can reuse our Actor model architecture
from actor_critic_model import Actor

# --- Hyperparameters for the 30-City Problem ---
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_CITIES = 30
BATCH_SIZE = 128
NUM_EPOCHS = 5000  # Give it the same number of epochs as the hybrid models
LEARNING_RATE = 1e-4


def calculate_tour_length_torch(cities, tour_indices):
    """Calculates the total length of a batch of tours using PyTorch."""
    batch_size, num_cities, _ = cities.shape
    tour_cities = torch.gather(cities, 1, tour_indices.unsqueeze(-1).expand(batch_size, num_cities, 2))
    tour_distances = (tour_cities - torch.roll(tour_cities, 1, dims=1)).norm(p=2, dim=2)
    return tour_distances.sum(dim=1)


def train_pure_rl():
    """
    Trains a pure RL model from scratch on 30-city problems.
    """
    print(f"--- Starting Pure RL Model Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # We only need the Actor model for this training run
    model = Actor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    avg_tour_lengths = []
    best_avg_length = float('inf')

    for epoch in range(NUM_EPOCHS):
        # 1. Generate a batch of random TSP problems (no Christofides ordering)
        cities_torch = torch.rand(BATCH_SIZE, NUM_CITIES, INPUT_DIM).to(device)

        # --- Standard REINFORCE Training Loop ---
        model.train()
        tour_indices, tour_log_probs, _ = model(cities_torch)  # We don't need the hidden state

        # Calculate reward (tour length)
        reward = calculate_tour_length_torch(cities_torch, tour_indices)

        # Calculate simple REINFORCE loss
        loss = (reward * tour_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_length = reward.mean().item()
        avg_tour_lengths.append(avg_length)

        # Save the best model found so far
        if avg_length < best_avg_length:
            best_avg_length = avg_length
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving model... **")
            torch.save(model.state_dict(), "pure_rl_model_30.pth")

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Avg Tour Length: {avg_length:.4f}")

    print("--- Pure RL Training Finished ---")
    print(f"The best model was saved with an average length of {best_avg_length:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Pure RL Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.savefig("pure_rl_learning_curve_30.png")
    plt.show()


if __name__ == '__main__':
    # Use a new seed to give this model its own fair chance
    torch.manual_seed(2025)
    train_pure_rl()