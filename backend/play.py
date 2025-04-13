import torch
import numpy as np
from utils.board import TicTacToeBoard
from models.tic_tac_toe_model import TicTacToeModel
from utils.feedback import FeedbackDataset
import os

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

def get_feedback():
    """Get feedback from the user about the last move"""
    while True:
        feedback = input("Was this a good move? (y/n/s to skip): ").lower()
        if feedback in ['y', 'n', 's', '']:
            return feedback
        print("Invalid input! Please enter 'y', 'n', or 's'.")

def play_game(feedback_mode=False):
    """
    Play a game of Tic Tac Toe against the trained model
    
    Args:
        feedback_mode: Whether to collect feedback on moves
    """
    # Load the model
    model = TicTacToeModel(num_blocks=3, hidden_size=128)
    model_path = 'models/tic_tac_toe_model_final.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    model.eval()
    
    # Create a feedback dataset if in feedback mode
    feedback_dataset = None
    if feedback_mode:
        feedback_dataset = FeedbackDataset()
        print(f"Feedback collection enabled. Current dataset has {feedback_dataset.get_stats()['total_examples']} examples.")
    
    # Create a new board
    board = TicTacToeBoard()
    game_history = []  # Store (state, move) pairs for review
    
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
        current_state = board.get_state()
        print_board(current_state)
        
        # Store current state for feedback collection
        current_state_flat = current_state.flatten()
        
        # Display board evaluation if available
        try:
            board_value = model.get_board_value(current_state)
            if board_value > 0:
                print(f"Model evaluation: X has advantage ({board_value:.2f})")
            elif board_value < 0:
                print(f"Model evaluation: O has advantage ({-board_value:.2f})")
            else:
                print(f"Model evaluation: Even position ({board_value:.2f})")
        except:
            pass
        
        if human_turn:
            print("Your turn!")
            while True:
                move = get_human_move()
                if move in board.available_moves():
                    # Record move for potential feedback
                    if feedback_mode:
                        game_history.append((current_state_flat.copy(), move))
                    
                    board.make_move(*move)
                    
                    # Optionally get feedback on your own move
                    if feedback_mode and input("Do you want to rate your own move? (y/n): ").lower() == 'y':
                        feedback = get_feedback()
                        if feedback == 'y':
                            feedback_dataset.add_positive_example(current_state_flat, move)
                        elif feedback == 'n':
                            feedback_dataset.add_negative_example(current_state_flat, move)
                    
                    break
                else:
                    print("Invalid move! That position is already taken.")
        else:
            print("Model's turn...")
            available_moves = board.available_moves()
            if available_moves:
                move = model.predict_move(board.get_state(), available_moves)
                
                # Record AI move for feedback
                if feedback_mode:
                    game_history.append((current_state_flat.copy(), move))
                
                board.make_move(*move)
                print(f"Model plays: {move[0]},{move[1]}")
                
                # Get feedback on the model's move
                if feedback_mode:
                    feedback = get_feedback()
                    if feedback == 'y':
                        feedback_dataset.add_positive_example(current_state_flat, move)
                    elif feedback == 'n':
                        feedback_dataset.add_negative_example(current_state_flat, move)
        
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
    
    # Review the game for feedback
    if feedback_mode:
        if input("Do you want to review the game moves and provide feedback? (y/n): ").lower() == 'y':
            review_game(game_history, board, feedback_dataset)
        
        # Save feedback data
        feedback_dataset.save_data()

def review_game(game_history, board, feedback_dataset):
    """
    Allow the user to review the game and provide feedback on specific moves
    
    Args:
        game_history: List of (state, move) tuples
        board: TicTacToeBoard instance
        feedback_dataset: FeedbackDataset for storing feedback
    """
    board.reset()
    print("\n----- GAME REVIEW -----")
    
    for i, (state, move) in enumerate(game_history):
        # Print the board state before the move
        print(f"\nMove {i+1}:")
        print_board(board.get_state())
        
        # Show the move that was made
        row, col = move
        print(f"Player {'X' if board.current_player == 1 else 'O'} plays: {row},{col}")
        
        # Make the move on the board
        board.make_move(row, col)
        
        # Ask for feedback if not already provided
        feedback = get_feedback()
        if feedback == 'y':
            feedback_dataset.add_positive_example(state, move)
        elif feedback == 'n':
            feedback_dataset.add_negative_example(state, move)
    
    print("\nFinal position:")
    print_board(board.get_state())
    print("----- END OF REVIEW -----")

def main_menu():
    """Display main menu and handle user choices"""
    while True:
        print("\n===== TIC TAC TOE MENU =====")
        print("1. Play against the model")
        print("2. Play and provide feedback")
        print("3. Train model with feedback")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            play_game(feedback_mode=False)
        elif choice == '2':
            play_game(feedback_mode=True)
        elif choice == '3':
            from train_with_feedback import train_with_human_feedback
            train_with_human_feedback()
        elif choice == '4':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()