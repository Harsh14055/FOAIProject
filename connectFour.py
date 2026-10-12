import pygame
import sys
import math
import random
import time

# Initialize Pygame
pygame.init()

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7

SQUARESIZE = 100
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

RADIUS = int(SQUARESIZE / 2 - 5)

# Colors
BLUE = (28, 107, 160)
BLACK = (20, 20, 20)
RED = (217, 83, 79)
YELLOW = (240, 173, 78)
WHITE = (255, 255, 255)
GREEN = (46, 204, 113)  # Vibrant green for winning line highlight

SCREEN = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Connect Four AI: Minimax with Alpha-Beta Pruning")
myfont = pygame.font.SysFont("monospace", 22, bold=True)
smallfont = pygame.font.SysFont("monospace", 18)

# Game Pieces
EMPTY = 0
PLAYER = 1
AI = 2

# Metrics for Report
last_nodes_evaluated = 0
last_time_taken = 0
current_algorithm = "Alpha-Beta Pruning"

def create_board():
    return [[EMPTY for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT)]

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == EMPTY

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == EMPTY:
            return r

def get_winning_coordinates(board, piece):
    # Returns a list of (row, col) tuples for the 4 connected pieces if a win exists, else []
    # Horizontal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return [(r, c), (r, c+1), (r, c+2), (r, c+3)]
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return [(r, c), (r+1, c), (r+2, c), (r+3, c)]
    # Positive Diagonal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return [(r, c), (r+1, c+1), (r+2, c+2), (r+3, c+3)]
    # Negative Diagonal
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return [(r, c), (r-1, c+1), (r-2, c+2), (r-3, c+3)]
    return []

def winning_move(board, piece):
    return len(get_winning_coordinates(board, piece)) > 0

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER if piece == AI else AI

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 80

    return score

def score_position(board, piece):
    score = 0
    center_array = [board[r][COLUMN_COUNT//2] for r in range(ROW_COUNT)]
    score += center_array.count(piece) * 3

    for r in range(ROW_COUNT):
        row_array = board[r]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [board[r][c] for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(get_valid_locations(board)) == 0

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

node_counter = 0

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player):
    global node_counter
    node_counter += 1

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI):
                return (None, 100000 + depth)
            elif winning_move(board, PLAYER):
                return (None, -100000 - depth)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI))

    if maximizing_player:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [row_item[:] for row_item in board]
            drop_piece(b_copy, row, col, AI)
            new_score = minimax_alpha_beta(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [row_item[:] for row_item in board]
            drop_piece(b_copy, row, col, PLAYER)
            new_score = minimax_alpha_beta(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def draw_board(board, message="", winning_coords=[]):
    SCREEN.fill(BLACK)
    
    # Draw top panel for stats and winner notifications
    pygame.draw.rect(SCREEN, BLACK, (0, 0, WIDTH, SQUARESIZE))
    
    if message:
        msg_text = myfont.render(message, True, WHITE)
        SCREEN.blit(msg_text, (10, 35))
    else:
        algo_text = smallfont.render(f"Algo: {current_algorithm} (Press 'T' toggle)", True, WHITE)
        nodes_text = smallfont.render(f"Nodes: {last_nodes_evaluated}", True, (100, 220, 100))
        time_text = smallfont.render(f"Time: {last_time_taken:.1f}ms", True, (100, 220, 100))
        
        SCREEN.blit(algo_text, (10, 12))
        SCREEN.blit(nodes_text, (10, 38))
        SCREEN.blit(time_text, (440, 38))

    # Draw grid slots
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(SCREEN, BLUE, (c*SQUARESIZE, (r+1)*SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(SCREEN, BLACK, (int(c*SQUARESIZE + SQUARESIZE/2), int((r+1)*SQUARESIZE + SQUARESIZE/2)), RADIUS)
    
    # Draw pieces (highlighting winning line in Green)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] != EMPTY:
                if (r, c) in winning_coords:
                    color = GREEN
                elif board[r][c] == PLAYER:
                    color = RED
                else:
                    color = YELLOW
                pygame.draw.circle(SCREEN, color, (int(c*SQUARESIZE + SQUARESIZE/2), int((r+1)*SQUARESIZE + SQUARESIZE/2)), RADIUS)
                
    pygame.display.update()

def main():
    global node_counter, last_nodes_evaluated, last_time_taken, current_algorithm
    board = create_board()
    game_over = False
    turn = PLAYER

    draw_board(board)

    while True:
        winning_coords = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION and not game_over and turn == PLAYER:
                posx = event.pos[0]
                draw_board(board)
                pygame.draw.circle(SCREEN, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and turn == PLAYER:
                posx = event.pos[0]
                col = int(posx // SQUARESIZE)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER)

                    player_win_coords = get_winning_coordinates(board, PLAYER)
                    if player_win_coords:
                        draw_board(board, "Player Wins! Press 'R' to Restart", player_win_coords)
                        game_over = True
                    elif len(get_valid_locations(board)) == 0:
                        draw_board(board, "It's a Draw! Press 'R' to Restart")
                        game_over = True
                    else:
                        turn = AI
                        draw_board(board)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    current_algorithm = "Standard Minimax" if current_algorithm == "Alpha-Beta Pruning" else "Alpha-Beta Pruning"
                    print(f"Switched algorithm to: {current_algorithm}")
                    if not game_over:
                        draw_board(board)

                if event.key == pygame.K_r:
                    board = create_board()
                    game_over = False
                    turn = PLAYER
                    last_nodes_evaluated = 0
                    last_time_taken = 0
                    draw_board(board)

        # AI Turn
        if turn == AI and not game_over:
            node_counter = 0
            start_time = time.time()
            
            col, _ = minimax_alpha_beta(board, 5, -math.inf, math.inf, True)
                
            end_time = time.time()
            last_nodes_evaluated = node_counter
            last_time_taken = (end_time - start_time) * 1000

            if col is not None and is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI)

                ai_win_coords = get_winning_coordinates(board, AI)
                if ai_win_coords:
                    draw_board(board, "AI Wins! Press 'R' to Restart", ai_win_coords)
                    game_over = True
                elif len(get_valid_locations(board)) == 0:
                    draw_board(board, "It's a Draw! Press 'R' to Restart")
                    game_over = True
                else:
                    draw_board(board)
                    turn = PLAYER

if __name__ == "__main__":
    main()