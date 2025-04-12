import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3x3 board with 3 possible states (one-hot encoded) = 27 inputs
        # Process: Transform board into a one-hot encoding where:
        # - Empty = [1,0,0]
        # - X = [0,1,0]
        # - O = [0,0,1]
        
        self.fc1 = nn.Linear(9, 64)  # Flattened 3x3 board to hidden layer
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)   # Output layer (probability for each position)
    
    def forward(self, x):
        # x shape: (batch_size, 9) where each value is -1, 0, or 1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Apply mask to ensure we only predict for valid moves
        return x

    def predict_move(self, board, valid_moves):
        """
        Predict the best move given the current board state
        Args:
            board: 3x3 numpy array with 0, 1, -1
            valid_moves: list of (row, col) tuples
        Returns:
            best move as (row, col)
        """
        # Convert board to tensor and flatten
        board_tensor = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logits = self(board_tensor)
        
        # Create a mask for valid moves
        valid_moves_mask = torch.zeros(9, dtype=torch.bool)
        for row, col in valid_moves:
            valid_moves_mask[row * 3 + col] = True
        
        # Apply mask and get the best move
        masked_logits = logits.squeeze(0).clone()
        masked_logits[~valid_moves_mask] = float('-inf')
        best_move_idx = torch.argmax(masked_logits).item()
        
        # Convert index back to (row, col)
        return best_move_idx // 3, best_move_idx % 3