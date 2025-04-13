import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization - implements skip connections
    to help with gradient flow and training of deeper networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Linear(channels, channels)
        self.batch_norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.2)  # Dropout to reduce overfitting
    
    def forward(self, x):
        residual = x
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x += residual  # Skip connection: add the original input
        return x

class TicTacToeModel(nn.Module):
    def __init__(self, num_blocks=3, hidden_size=128):
        """
        Improved Tic Tac Toe model with residual connections and two heads:
        1. Policy head: Predicts move probabilities
        2. Value head: Evaluates board positions
        
        Args:
            num_blocks: Number of residual blocks in the network
            hidden_size: Size of hidden layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
        # Input embedding layer - transforms board state into feature representation
        # Input is 9 cells from the board (flattened 3x3)
        self.embedding = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # Policy head - predicts move probabilities for each position
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 9)  # Output for each position on 3x3 board
        )
        
        # Value head - evaluates how good the current position is
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def _one_hot_encode(self, x):
        """
        Properly one-hot encode the input by creating a representation for:
        - Empty (0) -> [1,0,0]
        - X (1) -> [0,1,0]
        - O (-1) -> [0,0,1]
        
        However, we do this implicitly through embedding rather than expanding dimensions
        """
        # We handle this in the embedding layer directly without dimension expansion
        return x
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Tensor of shape (batch_size, 9) representing flattened board states
                  where each element is -1 (O), 0 (empty), or 1 (X)
        Returns:
            policy_logits: Logits for move probabilities of shape (batch_size, 9)
            value: Board evaluation of shape (batch_size, 1)
        """
        # Embedding
        x = self.embedding(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
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
            policy_logits, value = self(board_tensor)
        
        # Create a mask for valid moves
        valid_moves_mask = torch.zeros(9, dtype=torch.bool)
        for row, col in valid_moves:
            valid_moves_mask[row * 3 + col] = True
        
        # Apply mask and get the best move
        masked_logits = policy_logits.squeeze(0).clone()
        masked_logits[~valid_moves_mask] = float('-inf')
        best_move_idx = torch.argmax(masked_logits).item()
        
        # Convert index back to (row, col)
        return best_move_idx // 3, best_move_idx % 3
    
    def get_board_value(self, board):
        """
        Evaluate the current board state
        Args:
            board: 3x3 numpy array with 0, 1, -1
        Returns:
            Value between -1 (likely to lose) and 1 (likely to win)
        """
        # Convert board to tensor and flatten
        board_tensor = torch.tensor(board.flatten(), dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            _, value = self(board_tensor)
        
        return value.item()