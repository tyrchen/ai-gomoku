import numpy as np
import random
import pickle
import threading  # Import threading for AI computations
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for Progressbar
from multiprocessing import Pool, cpu_count
import os

# Game Logic
class Board:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)  # 0-empty, 1-player1, 2-player2
        self.center = self.size // 2

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

    def make_move(self, x, y, player):
        if self.board[y, x] == 0:
            self.board[y, x] = player
            return True
        return False

    def undo_move(self, x, y):
        self.board[y, x] = 0

    def is_full(self):
        return np.all(self.board != 0)

    def check_win(self, player):
        # Check horizontal, vertical, and diagonal lines for a win
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y, x] == player:
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

    def is_empty(self):
        return np.all(self.board == 0)

# AI Player with Minimax and Learning
class AIPlayer:
    def __init__(self, player_number, board_size=15, weights=None, max_depth=2):
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2
        self.max_depth = max_depth  # Depth of the minimax search
        self.board_size = board_size
        self.center = self.board_size // 2
        if weights:
            self.weights = weights
        else:
            # Initialize evaluation weights
            self.weights = {
                'five': 100000,
                'open_four': 10000,
                'four': 1000,
                'open_three': 500,
                'three': 100,
                'open_two': 50,
                'two': 10
            }
        # Positional weight matrix favoring center positions
        self.position_weights = self.create_position_weights()

    def create_position_weights(self):
        center = self.board_size / 2
        weights = np.zeros((self.board_size, self.board_size))
        for y in range(self.board_size):
            for x in range(self.board_size):
                # The further from the center, the lower the weight
                distance = np.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)
                weights[y, x] = 1 / (1 + distance)
        # Normalize weights to range between 1 and 2
        min_w = np.min(weights)
        max_w = np.max(weights)
        weights = 1 + (weights - min_w) / (max_w - min_w)
        return weights

    def save_weights(self, filename='ai_weights.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filename='ai_weights.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.weights = pickle.load(f)
        except FileNotFoundError:
            pass

    def choose_action(self, board):
        if board.is_empty():
            # Place the first move in the center
            return (board.center, board.center)
        _, move = self.minimax(board, self.max_depth, -float('inf'), float('inf'), True)
        return move

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_full():
            score = self.evaluate_board(board)
            return score, None
        if maximizing_player:
            max_eval = -float('inf')
            best_move = None
            for move in self.get_candidate_moves(board):
                x, y = move[1], move[0]
                board.make_move(x, y, self.player_number)
                if board.check_win(self.player_number):
                    board.undo_move(x, y)
                    return float('inf'), move
                eval, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo_move(x, y)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_candidate_moves(board):
                x, y = move[1], move[0]
                board.make_move(x, y, self.opponent_number)
                if board.check_win(self.opponent_number):
                    board.undo_move(x, y)
                    return -float('inf'), move
                eval, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo_move(x, y)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_board(self, board):
        # Evaluate the board from AI's perspective
        score = 0
        score += self.evaluate_player(board, self.player_number)
        score -= self.evaluate_player(board, self.opponent_number)
        return score

    def evaluate_player(self, board, player):
        total_score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for y in range(board.size):
            for x in range(board.size):
                if board.board[y, x] == player:
                    position_weight = self.position_weights[y, x]
                    for dx, dy in directions:
                        total_score += self.evaluate_direction(board, x, y, dx, dy, player) * position_weight
        return total_score

    def evaluate_direction(self, board, x, y, dx, dy, player):
        consecutive = 0
        open_ends = 0
        score = 0
        # Positive direction
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.board[ny, nx] == player:
                    consecutive += 1
                elif board.board[ny, nx] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break
        # Negative direction
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < board.size and 0 <= ny < board.size:
                if board.board[ny, nx] == player:
                    consecutive += 1
                elif board.board[ny, nx] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break
        if consecutive >= 5:
            score += self.weights['five']
        elif consecutive == 4 and open_ends == 2:
            score += self.weights['open_four']
        elif consecutive == 4 and open_ends == 1:
            score += self.weights['four']
        elif consecutive == 3 and open_ends == 2:
            score += self.weights['open_three']
        elif consecutive == 3 and open_ends == 1:
            score += self.weights['three']
        elif consecutive == 2 and open_ends == 2:
            score += self.weights['open_two']
        elif consecutive == 2 and open_ends == 1:
            score += self.weights['two']
        return score

    def get_candidate_moves(self, board):
        # Limit moves to positions near existing stones
        moves = set()
        for y in range(board.size):
            for x in range(board.size):
                if board.board[y, x] != 0:
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < board.size and 0 <= ny < board.size:
                                if board.board[ny, nx] == 0:
                                    moves.add((ny, nx))
        if not moves:
            return board.get_empty_positions()
        return list(moves)

    def adjust_weights(self, reward):
        # Adjust weights based on the reward
        for key in self.weights:
            self.weights[key] += reward * 0.01  # Learning rate

# Function to run a single self-training game (used for multiprocessing)
def run_training_game(args):
    try:
        ai_player1_weights, ai_player2_weights, board_size = args
        board = Board(size=board_size)
        ai_player1 = AIPlayer(player_number=1, board_size=board_size, weights=ai_player1_weights, max_depth=1)
        ai_player2 = AIPlayer(player_number=2, board_size=board_size, weights=ai_player2_weights, max_depth=1)
        current_player = ai_player1 if random.choice([True, False]) else ai_player2
        game_over = False
        while not game_over:
            move = current_player.choose_action(board)
            if move:
                board.make_move(move[1], move[0], current_player.player_number)
                if board.check_win(current_player.player_number):
                    # Adjust weights
                    if current_player == ai_player1:
                        ai_player1.adjust_weights(1)
                        ai_player2.adjust_weights(-1)
                    else:
                        ai_player1.adjust_weights(-1)
                        ai_player2.adjust_weights(1)
                    game_over = True
                elif board.is_full():
                    # Adjust weights for draw
                    ai_player1.adjust_weights(0.5)
                    ai_player2.adjust_weights(0.5)
                    game_over = True
                else:
                    current_player = ai_player1 if current_player == ai_player2 else ai_player2
            else:
                game_over = True
        return ai_player1.weights, ai_player2.weights
    except Exception as e:
        # Return the exception to the main process
        return e

# User Interface (UI)
class GomokuGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku")
        self.main_menu()

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("800x600")  # Set window size

        # Unbind any previous key bindings
        self.root.unbind('p')
        self.root.unbind('s')
        self.root.unbind('<Return>')
        self.root.unbind('b')

        # Create a frame to center the content
        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        tk.Label(frame, text="Welcome to Gomoku", font=("Helvetica", 24)).pack(pady=40)
        play_button = tk.Button(frame, text="Play against AI", command=self.play_vs_ai_menu, width=20, font=("Helvetica", 16))
        play_button.pack(pady=20)
        self.root.bind('p', lambda event: self.play_vs_ai_menu())

        train_button = tk.Button(frame, text="Self Training", command=self.start_self_training, width=20, font=("Helvetica", 16))
        train_button.pack(pady=20)
        self.root.bind('s', lambda event: self.start_self_training())

    def play_vs_ai_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("800x600")  # Set window size

        # Unbind previous key bindings
        self.root.unbind('p')
        self.root.unbind('s')

        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        tk.Label(frame, text="Enter board size (default 30):", font=("Helvetica", 16)).pack(pady=20)
        self.board_size_entry_ai = tk.Entry(frame, font=("Helvetica", 16))
        self.board_size_entry_ai.insert(0, "30")
        self.board_size_entry_ai.pack(pady=10)
        start_button = tk.Button(frame, text="Start Game", command=self.start_game_vs_ai, width=20, font=("Helvetica", 16))
        start_button.pack(pady=20)
        self.root.bind('<Return>', lambda event: self.start_game_vs_ai())

        back_button = tk.Button(frame, text="Back to Main Menu", command=self.main_menu, width=20, font=("Helvetica", 16))
        back_button.pack(pady=20)
        self.root.bind('b', lambda event: self.main_menu())

    def start_game_vs_ai(self):
        try:
            board_size = int(self.board_size_entry_ai.get())
            if board_size < 5 or board_size > 50:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid board size between 5 and 50.")
            return
        self.player_number = random.choice([1, 2])
        self.board = Board(size=board_size)
        self.ai_player = AIPlayer(player_number=2 if self.player_number == 1 else 1, board_size=self.board.size)
        self.ai_player.load_weights()
        self.is_player_turn = self.player_number == 1
        self.draw_board()
        if self.player_number != 1:
            self.ai_move()

    def start_self_training(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("800x600")  # Set window size

        # Unbind previous key bindings
        self.root.unbind('p')
        self.root.unbind('s')

        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        tk.Label(frame, text="Enter number of games for self-training:", font=("Helvetica", 16)).pack(pady=20)
        self.num_games_entry = tk.Entry(frame, font=("Helvetica", 16))
        self.num_games_entry.pack(pady=10)
        tk.Label(frame, text="Enter board size (e.g., 15):", font=("Helvetica", 16)).pack(pady=20)
        self.board_size_entry = tk.Entry(frame, font=("Helvetica", 16))
        self.board_size_entry.pack(pady=10)
        start_button = tk.Button(frame, text="Start Training", command=self.run_self_training, width=20, font=("Helvetica", 16))
        start_button.pack(pady=20)
        self.root.bind('<Return>', lambda event: self.run_self_training())

        back_button = tk.Button(frame, text="Back to Main Menu", command=self.main_menu, width=20, font=("Helvetica", 16))
        back_button.pack(pady=20)
        self.root.bind('b', lambda event: self.main_menu())

    def run_self_training(self):
        try:
            num_games = int(self.num_games_entry.get())
            board_size = int(self.board_size_entry.get())
            if board_size < 5 or board_size > 50:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers. Board size should be between 5 and 50.")
            return

        ai_weights_file = 'ai_weights.pkl'
        if os.path.exists(ai_weights_file):
            with open(ai_weights_file, 'rb') as f:
                ai_weights = pickle.load(f)
        else:
            ai_weights = None

        # Prepare arguments for multiprocessing
        args = [(ai_weights, ai_weights, board_size) for _ in range(num_games)]

        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Training Progress")
        tk.Label(self.progress_window, text="Training in progress...", font=("Helvetica", 16)).pack(pady=20)
        self.progress = ttk.Progressbar(self.progress_window, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=20)
        self.progress['maximum'] = num_games
        self.progress['value'] = 0

        # Make the progress window modal
        self.progress_window.grab_set()

        self.root.after(100, self.start_training_process, args, num_games, ai_weights_file)

    def start_training_process(self, args, num_games, ai_weights_file):
        print("start_training_process")
        self.pool = Pool(processes=cpu_count())
        self.results = []
        self.num_completed = 0
        self.error_occurred = False  # Flag to check for errors

        def collect_result(result):
            if isinstance(result, Exception):
                # Handle the exception
                messagebox.showerror("Error", f"An error occurred during training: {result}")
                self.pool.terminate()
                self.pool.join()
                self.progress_window.destroy()
                self.main_menu()
                self.error_occurred = True
                return

            self.results.append(result)
            self.num_completed += 1
            self.progress['value'] = self.num_completed
            # Update the progress bar
            self.progress_window.update_idletasks()
            print(f"num_completed: {self.num_completed}, num_games: {num_games}")
            if self.num_completed == num_games:
                self.pool.close()
                print("self.pool.close()")
                if not self.error_occurred:
                    self.finalize_training(ai_weights_file)

        for arg in args:
            self.pool.apply_async(run_training_game, args=(arg,), callback=collect_result)

        # Close the pool when all tasks are done
        self.pool.close()

    def finalize_training(self, ai_weights_file):
        print("finalize_training")
        # Aggregate the weights
        ai_player1_weights = self.results[0][0]
        ai_player2_weights = self.results[0][1]
        for weights_p1, weights_p2 in self.results[1:]:
            for key in ai_player1_weights:
                ai_player1_weights[key] += weights_p1[key]
                ai_player2_weights[key] += weights_p2[key]

        # Average the weights
        num_games = len(self.results)
        for key in ai_player1_weights:
            ai_player1_weights[key] /= num_games
            ai_player2_weights[key] /= num_games

        # Save the updated weights
        with open(ai_weights_file, 'wb') as f:
            pickle.dump(ai_player1_weights, f)

        # Close the progress window
        self.progress_window.destroy()
        messagebox.showinfo("Training Completed", f"Self-training of {num_games} games completed.")
        self.main_menu()

    def draw_board(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.geometry("800x800")  # Adjust window size

        canvas_size = min(800, 20 * self.board.size)
        self.cell_size = canvas_size // self.board.size
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)
        self.draw_grid()
        self.update_canvas()
        if self.player_number == 1:
            self.root.title("Your Turn")
        else:
            self.root.title("AI's Turn")

    def draw_grid(self):
        for i in range(self.board.size):
            # Vertical lines
            self.canvas.create_line(
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2,
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2 + (self.board.size - 1) * self.cell_size)
            # Horizontal lines
            self.canvas.create_line(
                self.cell_size // 2,
                self.cell_size // 2 + i * self.cell_size,
                self.cell_size // 2 + (self.board.size - 1) * self.cell_size,
                self.cell_size // 2 + i * self.cell_size)

    def update_canvas(self):
        self.canvas.delete("piece")
        for y in range(self.board.size):
            for x in range(self.board.size):
                piece = self.board.board[y, x]
                if piece != 0:
                    color = 'black' if piece == 1 else 'white'
                    center_x = self.cell_size // 2 + x * self.cell_size
                    center_y = self.cell_size // 2 + y * self.cell_size
                    radius = self.cell_size // 2 - 2
                    self.canvas.create_oval(
                        center_x - radius, center_y - radius,
                        center_x + radius, center_y + radius,
                        fill=color, tags="piece")

    def human_move(self, event):
        if not self.is_player_turn:
            return
        x = int(round((event.x - self.cell_size // 2) / self.cell_size))
        y = int(round((event.y - self.cell_size // 2) / self.cell_size))
        if 0 <= x < self.board.size and 0 <= y < self.board.size:
            if self.board.make_move(x, y, self.player_number):
                self.update_canvas()
                if self.board.check_win(self.player_number):
                    self.ai_player.adjust_weights(-1)
                    self.ai_player.save_weights()
                    messagebox.showinfo("Game Over", "You win!")
                    self.main_menu()
                    return
                elif self.board.is_full():
                    self.ai_player.adjust_weights(0.5)
                    self.ai_player.save_weights()
                    messagebox.showinfo("Game Over", "It's a draw!")
                    self.main_menu()
                    return
                else:
                    self.is_player_turn = False
                    self.ai_move()
        else:
            messagebox.showwarning("Invalid Move", "Please click within the board.")

    def ai_move(self):
        self.is_player_turn = False
        self.root.title("AI's Turn")
        self.root.update_idletasks()

        def compute_move():
            move = self.ai_player.choose_action(self.board)
            self.root.after(0, lambda: self.process_ai_move(move))

        threading.Thread(target=compute_move).start()

    def process_ai_move(self, move):
        if move:
            self.board.make_move(move[1], move[0], self.ai_player.player_number)
            self.update_canvas()
            if self.board.check_win(self.ai_player.player_number):
                self.ai_player.adjust_weights(1)
                self.ai_player.save_weights()
                messagebox.showinfo("Game Over", "AI wins!")
                self.main_menu()
                return
            elif self.board.is_full():
                self.ai_player.adjust_weights(0.5)
                self.ai_player.save_weights()
                messagebox.showinfo("Game Over", "It's a draw!")
                self.main_menu()
                return
        else:
            self.ai_player.adjust_weights(0.5)
            self.ai_player.save_weights()
            messagebox.showinfo("Game Over", "It's a draw!")
            self.main_menu()
            return
        self.is_player_turn = True
        self.root.title("Your Turn")

def main():
    root = tk.Tk()
    app = GomokuGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()
