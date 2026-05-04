# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SIMPLE SCRIPT: Train a copycat robot

This script:
1. Reads the recorded data
2. Trains a new robot to copy what the expert did
3. Saves the copycat robot
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


class SimpleDataset(Dataset):
    """Just a container for our recorded data."""
    def __init__(self, data_file):
        data = torch.load(data_file)
        self.obs = data["observations"]
        self.actions = data["actions"]
        print(f"[INFO] Loaded dataset: {len(self.obs)} examples")
        
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class CopycatRobot(nn.Module):
    """Simple neural network that learns to copy the expert."""
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.net(x)


def train():
    parser = argparse.ArgumentParser(description="Train copycat robot")
    parser.add_argument("--data", type=str, default="behaviour_cloning/expert_data.pt", help="Data file from collect_data.py")
    parser.add_argument("--epochs", type=int, default=50, help="How many times to go through the data")
    parser.add_argument("--batch_size", type=int, default=1024, help="How many examples at once")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="behaviour_cloning/bc_policy.pt", help="Where to save the trained robot")
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        print("Run this first: python collect_data.py --load_run YOUR_RUN_NAME")
        return
    
    # Load the recorded data
    dataset = SimpleDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create the copycat robot
    obs_size = dataset.obs.shape[1]
    action_size = dataset.actions.shape[1]
    
    print(f"[INFO] Robot sees: {obs_size} numbers")
    print(f"[INFO] Robot does: {action_size} numbers")
    
    copycat = CopycatRobot(obs_size, action_size)
    optimizer = torch.optim.Adam(copycat.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()  # "Make my actions match the expert's actions"
    
    # Training loop
    print(f"\n[INFO] Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        count = 0
        
        for obs_batch, action_batch in dataloader:
            # Copycat guesses what to do
            predicted_actions = copycat(obs_batch)
            
            # Check how wrong it was
            loss = loss_fn(predicted_actions, action_batch)
            
            # Learn from mistakes
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.6f}")
    
    # Save the trained copycat
    torch.save({
        "model": copycat.state_dict(),
        "obs_size": obs_size,
        "action_size": action_size,
    }, args.output)
    
    print(f"\n[SUCCESS] Copycat robot saved to: {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    train()
