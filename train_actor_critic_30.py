import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from actor_critic_model import Actor, Critic
from christofides import solve_tsp_christofides

# --- Hyperparameters ---
INPUT_DIM = 2
HIDDEN_DIM = 128
NUM_CITIES = 30
BATCH_SIZE = 128
# We don't need to run for the full 5000, just long enough to hit the plateau
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-4


def calculate_tour_length_torch(cities, tour_indices):
    """Calculates the total length of a batch of tours using PyTorch."""
    batch_size, num_cities, _ = cities.shape
    tour_cities = torch.gather(cities, 1, tour_indices.unsqueeze(-1).expand(batch_size, num_cities, 2))
    tour_distances = (tour_cities - torch.roll(tour_cities, 1, dims=1)).norm(p=2, dim=2)
    return tour_distances.sum(dim=1)


def train_actor_critic():
    """
    Trains the Actor-Critic model with logic to save the best-performing model.
    """
    print(f"--- Final Actor-Critic Training ({NUM_CITIES} Cities) ---")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    actor = Actor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    critic = Critic(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE)

    avg_tour_lengths = []

    # --- NEW: Keep track of the best score ---
    best_avg_length = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        cities_torch = torch.rand(BATCH_SIZE, NUM_CITIES, INPUT_DIM).to(device)

        actor.train()
        critic.train()

        tour_indices, tour_log_probs, final_hidden_state = actor(cities_torch)

        value = critic(final_hidden_state).squeeze()
        reward = calculate_tour_length_torch(cities_torch, tour_indices)

        advantage = reward - value.detach()
        actor_loss = (advantage * tour_log_probs).mean()
        critic_loss = nn.functional.mse_loss(value, reward)
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_length = reward.mean().item()
        avg_tour_lengths.append(avg_length)

        # --- NEW: Save-on-best logic ---
        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_epoch = epoch
            print(f"** New best average length at epoch {epoch + 1}: {best_avg_length:.4f} -> Saving models... **")
            torch.save(actor.state_dict(), "best_ac_actor_model_30.pth")
            torch.save(critic.state_dict(), "best_ac_critic_model_30.pth")

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Avg Tour Length: {avg_length:.4f}")

    print("\n--- Actor-Critic Training Finished ---")
    print(f"The best model was saved from epoch {best_epoch + 1} with an average length of {best_avg_length:.4f}")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(avg_tour_lengths)
    plt.title(f'Actor-Critic Model Learning Curve ({NUM_CITIES} Cities)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Tour Length')
    plt.grid(True)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Model (Epoch {best_epoch + 1})')
    plt.legend()
    plt.savefig("ac_final_learning_curve.png")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(12)
    train_actor_critic()