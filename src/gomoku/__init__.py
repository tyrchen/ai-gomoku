import numpy as np
import random
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

# AI Player with Heuristic
class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2

    def choose_action(self, board):
        # 1. Check for winning move
        move = self.find_winning_move(board, self.player_number)
        if move:
            return move
        # 2. Block opponent's winning move
        move = self.find_winning_move(board, self.opponent_number)
        if move:
            return move
        # 3. Choose best available move
        return self.best_move(board)

    def find_winning_move(self, board, player):
        for move in board.get_empty_positions():
            x, y = move[1], move[0]
            board.board[y, x] = player
            if board.check_win(player):
                board.board[y, x] = 0
                return move
            board.board[y, x] = 0
        return None

    def best_move(self, board):
        # Simple heuristic: choose a move adjacent to existing pieces
        empty_positions = board.get_empty_positions()
        random.shuffle(empty_positions)
        for move in empty_positions:
            x, y = move[1], move[0]
            if self.has_neighbor(board, x, y):
                return move
        # If no neighboring positions, pick random
        return random.choice(empty_positions)

    def has_neighbor(self, board, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < board.size and 0 <= ny < board.size:
                    if board.board[ny, nx] != 0:
                        return True
        return False

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
                        game_over = True
                    elif board.is_full():
                        game_over = True
                    else:
                        current_player = ai_player1 if current_player == ai_player2 else ai_player2
                else:
                    game_over = True
            print(f"Game {i+1}/{num_games} completed.")

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
                    messagebox.showinfo("Game Over", "You win!")
                    self.main_menu()
                elif self.board.is_full():
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
                messagebox.showinfo("Game Over", "AI wins!")
                self.main_menu()
            elif self.board.is_full():
                messagebox.showinfo("Game Over", "It's a draw!")
                self.main_menu()
        else:
            messagebox.showinfo("Game Over", "It's a draw!")
            self.main_menu()

def main():
    root = tk.Tk()
    app = GomokuGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()