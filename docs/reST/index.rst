import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)

# Screen Setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("OASIS: Beyond the Veil")
clock = pygame.time.Clock()

# Game State
class Player:
    def __init__(self, username):
        self.username = username
        self.avatar_image = pygame.Surface((50, 50))
        self.avatar_image.fill(CYAN)
        self.rect = self.avatar_image.get_rect(center=(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)))

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

def game_loop():
    running = True
    player = Player("Player1")  # This can be expanded to allow multiple players
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Movement Control (for demonstration purposes)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-5, 0)
        if keys[pygame.K_RIGHT]:
            player.move(5, 0)
        if keys[pygame.K_UP]:
            player.move(0, -5)
        if keys[pygame.K_DOWN]:
            player.move(0, 5)

        # Rendering
        screen.fill(BLACK)
        screen.blit(player.avatar_image, player.rect)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    game_loop()
