import torch
import numpy as np
from utils.board import TicTacToeBoard
from models.tic_tac_toe_model import TicTacToeModel

def print_board(board):
    """Print the tic tac toe board"""
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\n-------------")
    for i in range(3):
        row = ""
        for j in range(3):
            row += f"| {symbols[board[i, j]]} "
        print(row + "|")
        print("-------------")

def get_human_move():
    """Get a move from the human player"""
    while True:
        try:
            move = input("Enter your move (row,col), e.g., '1,2': ")
            row, col = map(int, move.split(','))
            if 0 <= row <= 2 and 0 <= col <= 2:
                return row, col
            else:
                print("Invalid move! Row and column must be between 0 and 2.")
        except:
            print("Invalid input! Please enter in the format 'row,col'")

def play_game():
    """Play a game of Tic Tac Toe against the trained model"""
    # Load the model
    model = TicTacToeModel()
    model.load_state_dict(torch.load('models/tic_tac_toe_model_final.pt'))
    model.eval()
    
    # Create a new board
    board = TicTacToeBoard()
    
    # Decide who goes first
    human_player = input("Do you want to play as X (goes first) or O? ").upper()
    if human_player == 'X':
        human_turn = True
        human_symbol = 1
    else:
        human_turn = False
        human_symbol = -1
    
    # Game loop
    while not board.game_over:
        # Display the current board
        print_board(board.get_state())
        
        if human_turn:
            print("Your turn!")
            while True:
                move = get_human_move()
                if move in board.available_moves():
                    board.make_move(*move)
                    break
                else:
                    print("Invalid move! That position is already taken.")
        else:
            print("Model's turn...")
            available_moves = board.available_moves()
            if available_moves:
                move = model.predict_move(board.get_state(), available_moves)
                board.make_move(*move)
                print(f"Model plays: {move[0]},{move[1]}")
        
        # Switch turns
        human_turn = not human_turn
    
    # Game over
    print_board(board.get_state())
    if board.winner == human_symbol:
        print("Congratulations! You won!")
    elif board.winner is None:
        print("It's a draw!")
    else:
        print("The model won!")

if __name__ == "__main__":
    play_game()