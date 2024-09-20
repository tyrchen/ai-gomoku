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

    def undo_move(self, x, y):
        self.board[y, x] = 0

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

# AI Player with Minimax and Learning
class AIPlayer:
    def __init__(self, player_number, weights=None):
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2
        self.max_depth = 2  # Depth of the minimax search
        if weights:
            self.weights = weights
        else:
            # Initialize evaluation weights
            self.weights = {
                'five': 100000,
                'open_four': 10000,
                'four': 1000,
                'open_three': 1000,
                'three': 100,
                'open_two': 100,
                'two': 10
            }

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
                    for dx, dy in directions:
                        total_score += self.evaluate_direction(board, x, y, dx, dy, player)
        return total_score

    def evaluate_direction(self, board, x, y, dx, dy, player):
        consecutive = 0
        open_ends = 0
        score = 0

        for i in range(5):
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

        if consecutive == 5:
            score += self.weights['five']
        elif consecutive == 4 and open_ends == 1:
            score += self.weights['open_four']
        elif consecutive == 4 and open_ends == 0:
            score += self.weights['four']
        elif consecutive == 3 and open_ends == 1:
            score += self.weights['open_three']
        elif consecutive == 3 and open_ends == 0:
            score += self.weights['three']
        elif consecutive == 2 and open_ends == 1:
            score += self.weights['open_two']
        elif consecutive == 2 and open_ends == 0:
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
        self.ai_player.load_weights()
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
        ai_player1.load_weights()
        ai_player2.load_weights()
        self.self_train(ai_player1, ai_player2, num_games, board)
        messagebox.showinfo("Training Completed", f"Self-training of {num_games} games completed.")
        self.main_menu()

    def self_train(self, ai_player1, ai_player2, num_games, board):
        for i in range(num_games):
            board.reset()
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
            print(f"Game {i+1}/{num_games} completed.")
        ai_player1.save_weights()
        ai_player2.save_weights()

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
                    center_x = 20 + x*40
                    center_y = 20 + y*40
                    self.canvas.create_oval(
                        center_x - 15, center_y - 15,
                        center_x + 15, center_y + 15,
                        fill=color, tags="piece")

    def human_move(self, event):
        x = round((event.x - 20) / 40)
        y = round((event.y - 20) / 40)
        if 0 <= x < self.board.size and 0 <= y < self.board.size:
            if self.board.make_move(x, y, self.player_number):
                self.update_canvas()
                if self.board.check_win(self.player_number):
                    self.ai_player.adjust_weights(-1)
                    self.ai_player.save_weights()
                    messagebox.showinfo("Game Over", "You win!")
                    self.main_menu()
                elif self.board.is_full():
                    self.ai_player.adjust_weights(0.5)
                    self.ai_player.save_weights()
                    messagebox.showinfo("Game Over", "It's a draw!")
                    self.main_menu()
                else:
                    self.ai_move()

    def ai_move(self):
        move = self.ai_player.choose_action(self.board)
        if move:
            self.board.make_move(move[1], move[0], self.ai_player.player_number)
            self.update_canvas()
            if self.board.check_win(self.ai_player.player_number):
                self.ai_player.adjust_weights(1)
                self.ai_player.save_weights()
                messagebox.showinfo("Game Over", "AI wins!")
                self.main_menu()
            elif self.board.is_full():
                self.ai_player.adjust_weights(0.5)
                self.ai_player.save_weights()
                messagebox.showinfo("Game Over", "It's a draw!")
                self.main_menu()
        else:
            self.ai_player.adjust_weights(0.5)
            self.ai_player.save_weights()
            messagebox.showinfo("Game Over", "It's a draw!")
            self.main_menu()

def main():
    root = tk.Tk()
    app = GomokuGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()