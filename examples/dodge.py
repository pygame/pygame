import pygame
import random
import sys

# Khởi tạo
pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Square Dodge")
clock = pygame.time.Clock()

# Màu
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Người chơi
player_size = 40
player = pygame.Rect(WIDTH//2 - player_size//2, HEIGHT - 60, player_size, player_size)
player_speed = 5

# Chướng ngại vật
enemy_size = 40
enemies = []
enemy_speed = 4
spawn_timer = 0

# Game loop
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Di chuyển người chơi
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player.left > 0:
        player.x -= player_speed
    if keys[pygame.K_RIGHT] and player.right < WIDTH:
        player.x += player_speed

    # Sinh chướng ngại vật
    spawn_timer += 1
    if spawn_timer > 30:
        x_pos = random.randint(0, WIDTH - enemy_size)
        enemies.append(pygame.Rect(x_pos, 0, enemy_size, enemy_size))
        spawn_timer = 0

    # Di chuyển chướng ngại vật
    for enemy in enemies[:]:
        enemy.y += enemy_speed
        if enemy.top > HEIGHT:
            enemies.remove(enemy)
        if enemy.colliderect(player):
            print("💥 Game Over!")
            pygame.quit()
            sys.exit()

    # Vẽ
    pygame.draw.rect(screen, RED, player)
    for enemy in enemies:
        pygame.draw.rect(screen, BLACK, enemy)

    pygame.display.flip()
    clock.tick(60)
