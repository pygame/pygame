import pygame
import sys
import time
import random

pygame.init()

# --- Paramètres principaux ---
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
FPS = 60

# --- Initialisation écran ---
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jeu style borne d'arcade")
clock = pygame.time.Clock()

# --- Joueurs ---
player_size = 50
player_speed = 5

# --- Classe Joueur ---
class Player:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, player_size, player_size)
        self.color = color
        self.paused_until = 0

    def move(self, keys, up, down, left, right):
        if time.time() < self.paused_until:
            return
        if keys[up]: self.rect.y -= player_speed
        if keys[down]: self.rect.y += player_speed
        if keys[left]: self.rect.x -= player_speed
        if keys[right]: self.rect.x += player_speed

    def draw(self, screen):
        # Effet style arcade: contour lumineux
        pygame.draw.rect(screen, WHITE, self.rect.inflate(6, 6))  # contour
        pygame.draw.rect(screen, self.color, self.rect)

# --- Obstacles ---
class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

def generate_obstacles(mode):
    obstacles = []
    count = 20 if mode == "solo" else 10
    for _ in range(count):
        x = random.randint(100, WIDTH-100)
        y = random.randint(100, HEIGHT-100)
        w = random.randint(20, 50)
        h = random.randint(20, 50)
        obstacles.append(Obstacle(x, y, w, h))
    return obstacles

# --- Sélection du mode ---
def select_mode():
    selecting = True
    font = pygame.font.SysFont('arial', 50, bold=True)
    while selecting:
        screen.fill(BLACK)
        # Style arcade: bordure autour du menu
        pygame.draw.rect(screen, YELLOW, (100, 100, 600, 400), 5)
        texts = ["1. Jouer seul", "2. Jouer 1v1", "3. Jouer contre l'IA"]
        for i, t in enumerate(texts):
            screen.blit(font.render(t, True, GREEN), (150, 150 + i*100))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: return "solo"
                if event.key == pygame.K_2: return "1v1"
                if event.key == pygame.K_3: return "IA"

# --- Collision avec obstacles ---
def check_obstacles(player, obstacles):
    for obs in obstacles:
        if player.rect.colliderect(obs.rect):
            player.paused_until = time.time() + 3

# --- Déplacement IA simple ---
def move_ai(ai_player, target_player):
    if time.time() < ai_player.paused_until:
        return
    if ai_player.rect.x < target_player.rect.x: ai_player.rect.x += player_speed
    if ai_player.rect.x > target_player.rect.x: ai_player.rect.x -= player_speed
    if ai_player.rect.y < target_player.rect.y: ai_player.rect.y += player_speed
    if ai_player.rect.y > target_player.rect.y: ai_player.rect.y -= player_speed

# --- Jeu principal ---
def game(mode):
    obstacles = generate_obstacles(mode)
    player1 = Player(100, HEIGHT//2, BLUE)
    player2 = None
    if mode == "1v1":
        player2 = Player(700, HEIGHT//2, GREEN)
    elif mode == "IA":
        player2 = Player(700, HEIGHT//2, YELLOW)

    finish_line = pygame.Rect(WIDTH-50, 0, 10, HEIGHT)

    running = True
    font = pygame.font.SysFont('arial', 50, bold=True)
    while running:
        screen.fill(BLACK)
        keys = pygame.key.get_pressed()

        # Déplacement
        player1.move(keys, pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d)
        if player2:
            if mode == "1v1":
                player2.move(keys, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)
            else:
                move_ai(player2, player1)

        # Vérification obstacles
        check_obstacles(player1, obstacles)
        if player2:
            check_obstacles(player2, obstacles)

        # Dessin style arcade
        pygame.draw.rect(screen, YELLOW, (0,0,WIDTH,HEIGHT), 10)  # cadre arcade
        player1.draw(screen)
        if player2: player2.draw(screen)
        for obs in obstacles: obs.draw(screen)
        pygame.draw.rect(screen, WHITE, finish_line)

        # Vérification ligne d'arrivée
        if player1.rect.colliderect(finish_line):
            msg = font.render("Joueur 1 gagne !", True, GREEN)
            screen.blit(msg, (WIDTH//2-150, HEIGHT//2))
            pygame.display.flip()
            pygame.time.delay(3000)
            return
        if player2 and player2.rect.colliderect(finish_line):
            msg = font.render("Joueur 2 gagne !", True, RED)
            screen.blit(msg, (WIDTH//2-150, HEIGHT//2))
            pygame.display.flip()
            pygame.time.delay(3000)
            return

        # Événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(FPS)

# --- Lancement ---
while True:
    mode = select_mode()
    game(mode)
