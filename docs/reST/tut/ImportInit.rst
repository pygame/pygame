import pygame
import random

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pacman & Dig Dug")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Define constants
PLAYER_SIZE = 30
PLAYER_SPEED = 5
ENEMY_SIZE = 30
ENEMY_SPEED = 3

# Define classes
class Player(pygame.sprite.Sprite):
    def __init__(self, color):
        super().__init__()
        self.image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.vx, self.vy = 0, 0

    def update(self):
        self.rect.x += self.vx
        self.rect.y += self.vy

class Enemy(pygame.sprite.Sprite):
    def __init__(self, color):
        super().__init__()
        self.image = pygame.Surface((ENEMY_SIZE, ENEMY_SIZE))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
        self.vx = random.choice([-ENEMY_SPEED, ENEMY_SPEED])
        self.vy = random.choice([-ENEMY_SPEED, ENEMY_SPEED])

    def update(self):
        self.rect.x += self.vx
        self.rect.y += self.vy

# Initialize sprites
all_sprites = pygame.sprite.Group()
players = pygame.sprite.Group()
enemies = pygame.sprite.Group()

player = Player(YELLOW)
enemy = Enemy(RED)

all_sprites.add(player, enemy)
players.add(player)
enemies.add(enemy)

# Main game loop
running = True
clock = pygame.time.Clock()
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update
    all_sprites.update()

    # Drawing
    WIN.fill(BLACK)
    all_sprites.draw(WIN)
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit the game
pygame.quit()
