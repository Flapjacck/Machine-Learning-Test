import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os

from utils.board import TicTacToeBoard
from models.tic_tac_toe_model import TicTacToeModel

# Ensure directories exist
os.makedirs('models/', exist_ok=True)
os.makedirs('data/', exist_ok=True)

# Create model and optimizer
model = TicTacToeModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training parameters
NUM_EPISODES = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
GAMMA = 0.99  # Discount factor

# Epsilon decay for exploration-exploitation tradeoff
epsilon = EPSILON_START

# Store game memories for replay
memories = []

def train_self_play():
    global epsilon
    win_count = 0
    
    for episode in tqdm(range(NUM_EPISODES)):
        board = TicTacToeBoard()
        game_memories = []
        
        while not board.game_over:
            state = board.get_state().flatten()
            available = board.available_moves()
            
            # Epsilon-greedy policy
            if random.random() < epsilon:
                # Exploration: random move
                move = random.choice(available)
            else:
                # Exploitation: use model
                board_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(board_tensor).squeeze(0)
                
                # Create a mask for valid moves
                valid_moves_mask = torch.zeros(9, dtype=torch.bool)
                for row, col in available:
                    valid_moves_mask[row * 3 + col] = True
                
                # Apply mask and get the best move
                masked_logits = logits.clone()
                masked_logits[~valid_moves_mask] = float('-inf')
                best_move_idx = torch.argmax(masked_logits).item()
                move = (best_move_idx // 3, best_move_idx % 3)
            
            # Save state and action
            old_state = state.copy()
            row, col = move
            
            # Make move
            board.make_move(row, col)
            
            # Save transition
            new_state = board.get_state().flatten()
            
            # Calculate reward
            reward = 0
            if board.game_over:
                if board.winner == 1:  # X wins
                    reward = 1
                    win_count += 1
                elif board.winner == -1:  # O wins
                    reward = -1
            
            # Store memory
            target_idx = row * 3 + col
            game_memories.append((old_state, target_idx, reward, new_state, board.game_over))
        
        # Add game memories to replay buffer
        memories.extend(game_memories)
        
        # Limit memory size
        if len(memories) > 10000:
            memories.pop(0)
        
        # Sample batch and train
        if len(memories) >= 128:
            batch = random.sample(memories, 128)
            train_batch(batch)
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Save model periodically
        if (episode + 1) % 1000 == 0:
            torch.save(model.state_dict(), f'models/tic_tac_toe_model_{episode+1}.pt')
            print(f"Episode {episode+1}, Win rate: {win_count/1000:.4f}")
            win_count = 0

def train_batch(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)
    
    # Current Q values
    current_q = model(states)
    current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q values
    with torch.no_grad():
        next_q = model(next_states)
        max_next_q = next_q.max(1)[0]
        target_q_values = rewards + GAMMA * max_next_q * (~dones)
    
    # Compute loss and update
    loss = criterion(current_q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def load_model_and_continue_training(model_path, additional_episodes=5000):
    """
    Load an existing model and continue training
    
    Args:
        model_path: Path to the saved model state dict
        additional_episodes: Number of additional episodes to train
    """
    global model, optimizer, NUM_EPISODES, epsilon
    
    # Load model
    model.load_state_dict(torch.load(model_path))
    
    # Reset optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for fine-tuning
    
    # Set training parameters
    NUM_EPISODES = additional_episodes
    epsilon = 0.3  # Start with lower epsilon for less exploration
    
    # Train model
    train_self_play()
    
    # Save continued model
    model_name = os.path.basename(model_path).split('.')[0]
    torch.save(model.state_dict(), f'models/{model_name}_continued.pt')
    print(f"Continued training completed. Model saved as {model_name}_continued.pt")

if __name__ == '__main__':
    """
    TRAINING GUIDE:
    
    Option 1: Train a new model from scratch
    -----------------------------------------
    This is the default behavior when you run:
        python train.py
    
    The script will:
    - Train for 10,000 episodes 
    - Save checkpoints every 1,000 episodes
    - Save the final model as 'tic_tac_toe_model_final.pt'
    
    Option 2: Continue training from a saved model
    ----------------------------------------------
    To continue training from a saved checkpoint:
    
    # Uncomment and modify the lines below:
    # model_path = 'models/tic_tac_toe_model_5000.pt'  # Path to existing model
    # load_model_and_continue_training(model_path, additional_episodes=5000)
    
    This will:
    - Load the specified model
    - Train for 5,000 more episodes with a lower learning rate
    - Save the new model as '{original_name}_continued.pt'
    
    Option 3: Customize training parameters
    ---------------------------------------
    To customize training hyperparameters, modify these variables at the top:
    - NUM_EPISODES: Total training episodes
    - EPSILON_START: Initial exploration rate (1.0 = 100% random moves)
    - EPSILON_END: Final exploration rate (0.1 = 10% random moves)
    - EPSILON_DECAY: Rate at which exploration decreases
    - GAMMA: Discount factor for future rewards (0.99 typical)
    
    Performance monitoring:
    ----------------------
    The script outputs the win rate every 1,000 episodes. A properly trained
    model should achieve win rates of at least 0.8 (80%) by episodes 5,000-10,000.
    
    After training:
    --------------
    Use 'play.py' to play against your trained model:
        python play.py
    """
    
    # Default behavior: Train from scratch
    train_self_play()
    
    # Save final model
    torch.save(model.state_dict(), 'models/tic_tac_toe_model_final.pt')