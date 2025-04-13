import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.tic_tac_toe_model import TicTacToeModel
from utils.feedback import FeedbackDataset
from utils.board import TicTacToeBoard

def train_with_human_feedback(base_model_path='models/tic_tac_toe_model_final.pt', 
                              output_model_path='models/tic_tac_toe_model_human_feedback.pt',
                              epochs=100,
                              learning_rate=0.0002,
                              batch_size=32,
                              feedback_weight=2.0):
    """
    Fine-tune a model with human feedback
    
    Args:
        base_model_path: Path to the base model to fine-tune
        output_model_path: Path to save the fine-tuned model
        epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        batch_size: Batch size for training
        feedback_weight: Weight to apply to human feedback examples vs self-play data
    """
    print("=== Training with Human Feedback ===")
    
    # Load feedback dataset
    feedback = FeedbackDataset()
    feedback_stats = feedback.get_stats()
    
    if feedback_stats['total_examples'] == 0:
        print("No feedback data found. Please play some games and provide feedback first.")
        return
    
    print(f"Loaded {feedback_stats['total_examples']} feedback examples "
          f"({feedback_stats['positive_examples']} positive, {feedback_stats['negative_examples']} negative)")
    
    # Load base model
    model = TicTacToeModel(num_blocks=3, hidden_size=128)
    
    if os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path))
        print(f"Loaded base model from {base_model_path}")
    else:
        print(f"Warning: Base model {base_model_path} not found. Starting with untrained model.")
    
    # Define loss and optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training parameters
    VALUE_WEIGHT = 0.5
    GAMMA = 0.99
    
    # Keep track of losses for visualization
    loss_history = []
    
    # Validation performance tracking
    validation_positions = [
        # Center opening
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
        # Corner opening
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), 
        # X in center, O in corner
        np.array([0, 0, -1, 0, 1, 0, 0, 0, 0]),
        # Block a potential win
        np.array([1, 1, 0, 0, -1, 0, 0, 0, 0]),
        # Go for a win
        np.array([1, 0, 0, 0, 1, -1, 0, -1, 0])
    ]
    
    # Track model predictions for validation positions
    validation_history = []
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        
        # Mix feedback data with some self-play data (if available)
        from train import memories
        
        # For each training step, we'll use a mix of feedback data and self-play memories
        num_batches = max(1, feedback_stats['total_examples'] // batch_size)
        
        for _ in range(num_batches):
            # Get a batch of human feedback data
            feedback_batch = feedback.get_feedback_batch(batch_size)
            
            if not feedback_batch:
                continue
                
            # Convert feedback batch to tensors
            fb_states, fb_actions, fb_rewards = [], [], []
            for state, action, reward in feedback_batch:
                fb_states.append(state)
                fb_actions.append(action)
                fb_rewards.append(reward)
            
            fb_states = torch.tensor(np.array(fb_states), dtype=torch.float32)
            fb_actions = torch.tensor(fb_actions, dtype=torch.long)
            fb_rewards = torch.tensor(fb_rewards, dtype=torch.float32)
            
            # Forward pass
            policy_logits, value = model(fb_states)
            
            # Policy loss (how well does the model predict the human's preferred moves)
            policy_loss = policy_criterion(policy_logits, nn.functional.one_hot(fb_actions, 9).float())
            
            # Value loss (the model should learn to value positions according to feedback)
            value_loss = value_criterion(value.squeeze(1), fb_rewards)
            
            # Combined loss with higher weight for feedback
            loss = feedback_weight * (policy_loss + VALUE_WEIGHT * value_loss)
            
            # Get some self-play data if available
            if hasattr(memories, '__len__') and len(memories) > batch_size:
                batch = random.sample(memories, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.bool)
                
                # Current policy and value predictions
                sp_policy_logits, sp_value = model(states)
                
                # Calculate policy loss
                sp_policy_loss = policy_criterion(sp_policy_logits, nn.functional.one_hot(actions, 9).float())
                
                # Target values
                with torch.no_grad():
                    model.eval()
                    _, next_value = model(next_states)
                    model.train()
                    target_values = rewards + GAMMA * next_value.squeeze(1) * (~dones)
                
                # Value loss
                sp_value_loss = value_criterion(sp_value.squeeze(1), target_values)
                
                # Add self-play loss (with lower weight)
                loss += sp_policy_loss + VALUE_WEIGHT * sp_value_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Record average loss for this epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Evaluate model on validation positions every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            validation_predictions = []
            
            for position in validation_positions:
                # Convert to tensor
                pos_tensor = torch.tensor(position, dtype=torch.float32).unsqueeze(0)
                
                # Get model predictions
                with torch.no_grad():
                    policy_logits, _ = model(pos_tensor)
                
                # Get best move
                board = np.array(position).reshape(3, 3)
                available = [(i//3, i%3) for i in range(9) if position[i] == 0]
                
                # Create a mask for valid moves
                valid_moves_mask = torch.zeros(9, dtype=torch.bool)
                for row, col in available:
                    valid_moves_mask[row * 3 + col] = True
                
                # Apply mask
                masked_logits = policy_logits.squeeze(0).clone()
                masked_logits[~valid_moves_mask] = float('-inf')
                best_move_idx = torch.argmax(masked_logits).item()
                
                validation_predictions.append((best_move_idx // 3, best_move_idx % 3))
            
            validation_history.append(validation_predictions)
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model fine-tuned with human feedback saved to {output_model_path}")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss with Human Feedback')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('data/human_feedback_training_loss.png')
    
    # Visualize how model predictions changed for validation positions
    if len(validation_history) > 1:
        # Print info about how the model's predictions changed
        initial_predictions = validation_history[0]
        final_predictions = validation_history[-1]
        
        print("\nChanges in model predictions for validation positions:")
        for i, (initial, final) in enumerate(zip(initial_predictions, final_predictions)):
            print(f"Position {i+1}: {initial} -> {final}")
    
    return model

if __name__ == "__main__":
    train_with_human_feedback()