import numpy as np

class TicTacToeBoard:
    def __init__(self):
        # 3x3 board represented as 0 (empty), 1 (X), -1 (O)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # X starts
        self.game_over = False
        self.winner = None
    
    def make_move(self, row, col):
        """Make a move on the board"""
        if self.game_over or self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        
        # Check for win or draw
        if self._check_winner():
            self.game_over = True
            self.winner = self.current_player
        elif np.all(self.board != 0):  # Board is full
            self.game_over = True
        
        # Switch player
        self.current_player = -self.current_player
        return True
    
    def _check_winner(self):
        """Check if current player has won"""
        # Check rows, columns and diagonals
        for i in range(3):
            # Check rows and columns
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return True
        
        # Check diagonals
        if abs(np.sum(np.diag(self.board))) == 3 or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        
        return False
    
    def get_state(self):
        """Return the board state as a numpy array"""
        return self.board.copy()
    
    def available_moves(self):
        """Return list of available move positions as (row, col) tuples"""
        if self.game_over:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def reset(self):
        """Reset the board to initial state"""
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None