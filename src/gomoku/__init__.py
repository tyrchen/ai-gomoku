import numpy as np
import random
import pickle
import tkinter as tk
from tkinter import messagebox

# Game Logic
class Board:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0-empty, 1-player1, 2-player2

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

    def make_move(self, x, y, player):
        if self.board[y, x] == 0:
            self.board[y, x] = player
            return True
        return False

    def is_full(self):
        return np.all(self.board != 0)

    def check_win(self, player):
        # Check horizontal, vertical, and diagonal lines for a win
        for y in range(self.size):
            for x in range(self.size):
                if self.check_line(x, y, 1, 0, player) \
                   or self.check_line(x, y, 0, 1, player) \
                   or self.check_line(x, y, 1, 1, player) \
                   or self.check_line(x, y, 1, -1, player):
                    return True
        return False

    def check_line(self, x, y, dx, dy, player):
        count = 0
        for i in range(5):
            nx, ny = x + dx*i, y + dy*i
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.board[ny, nx] == player:
                    count += 1
                else:
                    break
            else:
                break
        return count == 5

    def get_empty_positions(self):
        return list(zip(*np.where(self.board == 0)))

# AI Player with Learning Capability
class AIPlayer:
    def __init__(self, player_number, q_table=None):
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2
        self.q_table = q_table if q_table is not None else {}
        self.epsilon = 0.1  # Exploration rate

    def get_state_key(self, board):
        return tuple(board.board.flatten())

    def choose_action(self, board):
        state_key = self.get_state_key(board)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        empty_positions = board.get_empty_positions()
        if empty_positions:
            # Exploration vs. Exploitation
            if random.random() < self.epsilon:  # Exploration
                move = random.choice(empty_positions)
            else:
                # Exploitation: choose the best known move
                q_values = self.q_table[state_key]
                if q_values:
                    # Get the action with the highest Q-value
                    move = max(q_values, key=q_values.get)
                    if move not in empty_positions:
                        move = random.choice(empty_positions)
                else:
                    move = random.choice(empty_positions)
            return move
        else:
            return None

    def update_q_table(self, history, reward):
        # Simple Q-learning update
        learning_rate = 0.1
        discount_factor = 0.9
        next_max = 0
        for state, action in reversed(history):
            state_key = state
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0
            # Q-learning formula
            old_value = self.q_table[state_key][action]
            self.q_table[state_key][action] = old_value + learning_rate * (discount_factor * next_max - old_value)
            next_max = max(self.q_table[state_key].values())
        # Update the first move with the final reward
        first_state, first_action = history[-1]
        self.q_table[first_state][first_action] += learning_rate * (reward - self.q_table[first_state][first_action])

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

# Self-Training Mechanism
def self_train(ai_player1, ai_player2, num_games, board, q_table_file):
    for i in range(num_games):
        board.reset()
        history_p1 = []
        history_p2 = []
        current_player = ai_player1
        other_player = ai_player2
        game_over = False
        while not game_over:
            move = current_player.choose_action(board)
            if move:
                board.make_move(move[1], move[0], current_player.player_number)
                state_key = current_player.get_state_key(board)
                if current_player.player_number == ai_player1.player_number:
                    history_p1.append((state_key, move))
                else:
                    history_p2.append((state_key, move))
                if board.check_win(current_player.player_number):
                    # Current player wins
                    current_player.update_q_table(history_p1 if current_player == ai_player1 else history_p2, reward=1)
                    other_player.update_q_table(history_p2 if other_player == ai_player1 else history_p1, reward=-1)
                    game_over = True
                elif board.is_full():
                    # Draw
                    current_player.update_q_table(history_p1 if current_player == ai_player1 else history_p2, reward=0)
                    other_player.update_q_table(history_p2 if other_player == ai_player1 else history_p1, reward=0)
                    game_over = True
                else:
                    # Switch players
                    current_player, other_player = other_player, current_player
            else:
                # No moves left
                game_over = True
        print(f"Game {i+1}/{num_games} completed.")
    # Save Q-tables
    ai_player1.save_q_table(q_table_file)
    ai_player2.save_q_table(q_table_file)

# User Interface (UI)
class GomokuGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku")
        self.main_menu()

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Welcome to Gomoku", font=("Helvetica", 16)).pack(pady=20)
        tk.Button(self.root, text="Play against AI", command=self.start_game_vs_ai, width=20).pack(pady=10)
        tk.Button(self.root, text="Self Training", command=self.start_self_training, width=20).pack(pady=10)

    def start_game_vs_ai(self):
        self.player_number = random.choice([1, 2])
        self.board = Board()
        self.ai_player = AIPlayer(player_number=2 if self.player_number == 1 else 1)
        self.ai_player.load_q_table('q_table.pkl')
        self.history = []
        self.draw_board()
        if self.player_number != 1:
            self.ai_move()

    def start_self_training(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        tk.Label(self.root, text="Enter number of games for self-training:", font=("Helvetica", 12)).pack(pady=10)
        self.num_games_entry = tk.Entry(self.root)
        self.num_games_entry.pack(pady=5)
        tk.Button(self.root, text="Start Training", command=self.run_self_training).pack(pady=10)

    def run_self_training(self):
        try:
            num_games = int(self.num_games_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")
            return
        board = Board()
        ai_player1 = AIPlayer(player_number=1)
        ai_player2 = AIPlayer(player_number=2)
        ai_player1.load_q_table('q_table.pkl')
        ai_player2.load_q_table('q_table.pkl')
        self_train(ai_player1, ai_player2, num_games, board, 'q_table.pkl')
        messagebox.showinfo("Training Completed", f"Self-training of {num_games} games completed.")
        self.main_menu()

    def draw_board(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)
        self.draw_grid()
        self.update_canvas()

    def draw_grid(self):
        for i in range(self.board.size):
            # Vertical lines
            self.canvas.create_line(20 + i*40, 20, 20 + i*40, 580)
            # Horizontal lines
            self.canvas.create_line(20, 20 + i*40, 580, 20 + i*40)

    def update_canvas(self):
        self.canvas.delete("piece")
        for y in range(self.board.size):
            for x in range(self.board.size):
                piece = self.board.board[y, x]
                if piece != 0:
                    color = 'black' if piece == 1 else 'white'
                    self.canvas.create_oval(10 + x*40, 10 + y*40, 30 + x*40, 30 + y*40, fill=color, tags="piece")

    def human_move(self, event):
        x = (event.x - 20) // 40
        y = (event.y - 20) // 40
        if 0 <= x < self.board.size and 0 <= y < self.board.size:
            if self.board.make_move(x, y, self.player_number):
                self.update_canvas()
                state_key = tuple(self.board.board.flatten())
                self.history.append((state_key, (y, x)))
                if self.board.check_win(self.player_number):
                    self.ai_player.update_q_table(self.history, reward=-1)
                    messagebox.showinfo("Game Over", "You win!")
                    self.ai_player.save_q_table('q_table.pkl')
                    self.main_menu()
                elif self.board.is_full():
                    self.ai_player.update_q_table(self.history, reward=0)
                    messagebox.showinfo("Game Over", "It's a draw!")
                    self.ai_player.save_q_table('q_table.pkl')
                    self.main_menu()
                else:
                    self.ai_move()

    def ai_move(self):
        move = self.ai_player.choose_action(self.board)
        if move:
            self.board.make_move(move[1], move[0], self.ai_player.player_number)
            self.update_canvas()
            state_key = tuple(self.board.board.flatten())
            self.history.append((state_key, move))
            if self.board.check_win(self.ai_player.player_number):
                self.ai_player.update_q_table(self.history, reward=1)
                messagebox.showinfo("Game Over", "AI wins!")
                self.ai_player.save_q_table('q_table.pkl')
                self.main_menu()
            elif self.board.is_full():
                self.ai_player.update_q_table(self.history, reward=0)
                messagebox.showinfo("Game Over", "It's a draw!")
                self.ai_player.save_q_table('q_table.pkl')
                self.main_menu()
        else:
            self.ai_player.update_q_table(self.history, reward=0)
            messagebox.showinfo("Game Over", "It's a draw!")
            self.ai_player.save_q_table('q_table.pkl')
            self.main_menu()

def main():
    root = tk.Tk()
    app = GomokuGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()
