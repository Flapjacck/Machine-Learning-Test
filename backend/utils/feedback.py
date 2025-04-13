import json
import os
import numpy as np

class FeedbackDataset:
    """
    Class to store and manage human feedback on model moves.
    This dataset stores positive and negative examples that can be used to fine-tune the model.
    """
    def __init__(self, file_path='data/human_feedback.json'):
        self.file_path = file_path
        self.positive_examples = []  # Good moves with high reward
        self.negative_examples = []  # Bad moves with negative reward
        
        # Load existing data if file exists
        self.load_data()
    
    def load_data(self):
        """Load feedback data from file if it exists"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.positive_examples = data.get('positive', [])
                    self.negative_examples = data.get('negative', [])
                print(f"Loaded {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")
            except Exception as e:
                print(f"Error loading feedback data: {e}")
    
    def save_data(self):
        """Save feedback data to file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        data = {
            'positive': self.positive_examples,
            'negative': self.negative_examples
        }
        
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")
        except Exception as e:
            print(f"Error saving feedback data: {e}")
    
    def add_positive_example(self, state, move):
        """
        Add a positive example (good move)
        
        Args:
            state: Board state as a flattened list (before the move)
            move: (row, col) tuple or move index (0-8)
        """
        # Convert move to index if it's a tuple
        if isinstance(move, tuple):
            move_idx = move[0] * 3 + move[1]
        else:
            move_idx = move
            
        # Convert numpy array to list for JSON serialization
        if isinstance(state, np.ndarray):
            state = state.tolist()
            
        example = {
            'state': state,
            'action': move_idx,
            'reward': 1.0  # High reward for good moves
        }
        
        self.positive_examples.append(example)
        print("Added positive example!")
    
    def add_negative_example(self, state, move):
        """
        Add a negative example (bad move)
        
        Args:
            state: Board state as a flattened list (before the move)
            move: (row, col) tuple or move index (0-8)
        """
        # Convert move to index if it's a tuple
        if isinstance(move, tuple):
            move_idx = move[0] * 3 + move[1]
        else:
            move_idx = move
            
        # Convert numpy array to list for JSON serialization
        if isinstance(state, np.ndarray):
            state = state.tolist()
            
        example = {
            'state': state,
            'action': move_idx,
            'reward': -0.5  # Negative reward for bad moves
        }
        
        self.negative_examples.append(example)
        print("Added negative example!")
    
    def get_feedback_batch(self, batch_size=32):
        """
        Get a balanced batch of feedback examples for training
        
        Args:
            batch_size: Size of the batch to return
        
        Returns:
            List of (state, action, reward) tuples
        """
        batch = []
        
        # Determine how many examples to take from each category
        pos_size = min(len(self.positive_examples), batch_size // 2)
        neg_size = min(len(self.negative_examples), batch_size - pos_size)
        
        # Get random samples from positive and negative examples
        if pos_size > 0:
            pos_indices = np.random.choice(len(self.positive_examples), pos_size, replace=False)
            for idx in pos_indices:
                example = self.positive_examples[idx]
                batch.append((example['state'], example['action'], example['reward']))
        
        if neg_size > 0:
            neg_indices = np.random.choice(len(self.negative_examples), neg_size, replace=False)
            for idx in neg_indices:
                example = self.negative_examples[idx]
                batch.append((example['state'], example['action'], example['reward']))
        
        return batch
    
    def get_stats(self):
        """Return statistics about the feedback dataset"""
        return {
            'positive_examples': len(self.positive_examples),
            'negative_examples': len(self.negative_examples),
            'total_examples': len(self.positive_examples) + len(self.negative_examples)
        }