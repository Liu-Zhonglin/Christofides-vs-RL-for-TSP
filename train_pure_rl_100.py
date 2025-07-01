import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# We can reuse our Actor model architecture
from actor_critic_model import Actor

# --- Hyperparameters for the 100-City Problem ---
# These are now consistent with the 30-city experiments
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_CITIES = 100  # <-- The main change for this experiment
BATCH_SIZE = 128
NUM_EPOCHS = 5000  # <-- Corrected to match 30-city runs
LEARNING_RATE = 1e-4


def calculate_tour_length_torch(cities, tour_indices):
    """Calculates the total length of a batch of tours using PyTorch."""
    batch_size, num_cities, _ = cities.shape
    tour_cities = torch.gather(cities, 1, tour_indices.unsqueeze(-1).expand(batch_size, num_cities, 2))
    tour_distances = (tour_cities - torch.roll(tour_cities, 1, dims=1)).norm(p=2, dim=2)
    return tour_distances.sum(dim=1)


def train_pure_rl_100():
    """
    Trains a pure RL model from scratch on 100-city problems.
    """
    print(f"--- Starting Pure RL Model Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Actor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    avg_tour_lengths = []
    best_avg_length = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # Generate a batch of random 100-city problems
        cities_torch = torch.rand(BATCH_SIZE, NUM_CITIES, INPUT_DIM).to(device)

        model.train()
        tour_indices, tour_log_probs, _ = model(cities_torch)

        reward = calculate_tour_length_torch(cities_torch, tour_indices)

        loss = (reward * tour_log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_length = reward.mean().item()
        avg_tour_lengths.append(avg_length)

        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_epoch = epoch
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving model... **")
            torch.save(model.state_dict(), "pure_rl_model_100.pth")

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Avg Tour Length: {avg_length:.4f}")

    print("\n--- Pure RL (100 Cities) Training Finished ---")
    print(f"The best model was saved from epoch {best_epoch + 1} with an average length of {best_avg_length:.4f}")

    # Plot and save the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Pure RL Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.legend()
    plt.savefig("pure_rl_learning_curve_100.png")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(300)
    train_pure_rl_100()
