
import pygame
import sys

pygame.init()

# Pencere ayarları
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neon Dash - Python Mini Game")
clock = pygame.time.Clock()

# Renkler
BG = (18, 22, 55)
CYAN = (70, 245, 255)
WHITE = (255, 255, 255)
PINK = (255, 65, 130)
YELLOW = (255, 230, 80)
DARK = (30, 42, 85)

font = pygame.font.SysFont("Arial", 28, bold=True)
big_font = pygame.font.SysFont("Arial", 54, bold=True)

GROUND_Y = 410
PLAYER_SIZE = 38
GRAVITY = 0.8
JUMP_POWER = -14
SPEED = 6

def reset_game():
    return {
        "player_y": GROUND_Y - PLAYER_SIZE,
        "velocity_y": 0,
        "on_ground": True,
        "obstacles": [
            pygame.Rect(500, GROUND_Y - 38, 38, 38),
            pygame.Rect(760, GROUND_Y - 38, 38, 38),
            pygame.Rect(1020, GROUND_Y - 38, 38, 38),
            pygame.Rect(1300, GROUND_Y - 38, 38, 38),
            pygame.Rect(1580, GROUND_Y - 38, 38, 38),
            pygame.Rect(1870, GROUND_Y - 38, 38, 38),
        ],
        "score": 0,
        "game_over": False,
        "world_x": 0,
    }

game = reset_game()

def draw_spike(x, y, size):
    points = [
        (x, y + size),
        (x + size // 2, y),
        (x + size, y + size)
    ]
    pygame.draw.polygon(screen, PINK, points)
    pygame.draw.polygon(screen, WHITE, points, 2)

def draw_player(x, y):
    rect = pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE)
    pygame.draw.rect(screen, CYAN, rect, border_radius=5)
    pygame.draw.rect(screen, WHITE, rect, 3, border_radius=5)

    # Göz detayları
    pygame.draw.rect(screen, BG, (x + 9, y + 10, 6, 6))
    pygame.draw.rect(screen, BG, (x + 24, y + 10, 6, 6))
    return rect

def draw_background():
    screen.fill(BG)

    # Neon ızgara
    offset = int(game["world_x"] * 0.4) % 50
    for x in range(-offset, WIDTH, 50):
        pygame.draw.line(screen, (30, 45, 85), (x, 0), (x, GROUND_Y), 1)

    for y in range(0, GROUND_Y, 50):
        pygame.draw.line(screen, (30, 45, 85), (0, y), (WIDTH, y), 1)

    # Zemin
    pygame.draw.rect(screen, DARK, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.line(screen, CYAN, (0, GROUND_Y), (WIDTH, GROUND_Y), 4)

def draw_game_over():
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, (0, 0))

    title = big_font.render("OYUN BİTTİ!", True, PINK)
    score_text = font.render(f"Skor: {game['score']}", True, WHITE)
    restart_text = font.render("R tuşu: Yeniden başlat", True, CYAN)

    screen.blit(title, title.get_rect(center=(WIDTH // 2, 190)))
    screen.blit(score_text, score_text.get_rect(center=(WIDTH // 2, 250)))
    screen.blit(restart_text, restart_text.get_rect(center=(WIDTH // 2, 305)))

running = True

while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if game["game_over"] and event.key == pygame.K_r:
                game = reset_game()

            if event.key in (pygame.K_SPACE, pygame.K_UP):
                if game["on_ground"] and not game["game_over"]:
                    game["velocity_y"] = JUMP_POWER
                    game["on_ground"] = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if game["on_ground"] and not game["game_over"]:
                game["velocity_y"] = JUMP_POWER
                game["on_ground"] = False

    draw_background()

    player_x = 150

    if not game["game_over"]:
        # Yerçekimi ve zıplama
        game["velocity_y"] += GRAVITY
        game["player_y"] += game["velocity_y"]

        if game["player_y"] >= GROUND_Y - PLAYER_SIZE:
            game["player_y"] = GROUND_Y - PLAYER_SIZE
            game["velocity_y"] = 0
            game["on_ground"] = True

        # Dünya ilerler
        game["world_x"] += SPEED

        player_rect = draw_player(player_x, int(game["player_y"]))

        # Engelleri çiz ve çarpışmayı kontrol et
        for obstacle in game["obstacles"]:
            screen_x = obstacle.x - game["world_x"]

            if -50 < screen_x < WIDTH + 50:
                draw_spike(screen_x, obstacle.y, obstacle.width)

                spike_rect = pygame.Rect(
                    screen_x + 8,
                    obstacle.y + 10,
                    obstacle.width - 16,
                    obstacle.height - 10
                )

                if player_rect.colliderect(spike_rect):
                    game["game_over"] = True

            if screen_x + obstacle.width < player_x:
                # Her engel geçildiğinde puan kazan
                if not obstacle.width == -1:
                    game["score"] += 1
                    obstacle.width = -1

        # İlerleme göstergesi
        progress = min(100, int(game["world_x"] / 2200 * 100))
        progress_text = font.render(f"İlerleme: {progress}%", True, WHITE)
        screen.blit(progress_text, (20, 20))

        score_text = font.render(f"Skor: {game['score']}", True, YELLOW)
        screen.blit(score_text, (20, 55))

    else:
        draw_player(player_x, int(game["player_y"]))
        draw_game_over()

    pygame.display.flip()

pygame.quit()
sys.exit()
