import pygame
import random

# Pygame inicializálás
pygame.init()

# Képernyő beállítások
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Lövöldözős Játék")

# Színek
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Játékos beállításai
player_size = 50
player_x = WIDTH // 2
player_y = HEIGHT - 70
player_speed = 5

# Lövedékek
bullets = []
bullet_speed = 7
bullet_size = 5

# Ellenségek
enemy_size = 50
enemy_speed = 2
enemies = []

# Betűtípus
font = pygame.font.Font(None, 36)

# Játékóra
clock = pygame.time.Clock()

# Játékos rajzolása
def draw_player(x, y):
    pygame.draw.rect(screen, BLUE, (x, y, player_size, player_size))

# Lövedékek rajzolása
def draw_bullets():
    for bullet in bullets:
        pygame.draw.rect(screen, RED, (bullet[0], bullet[1], bullet_size, bullet_size))

# Ellenségek rajzolása
def draw_enemies():
    for enemy in enemies:
        pygame.draw.rect(screen, WHITE, (enemy[0], enemy[1], enemy_size, enemy_size))

# Fő ciklus
running = True
score = 0

while running:
    screen.fill((0, 0, 0))  # Háttér törlése

    # Eseménykezelés
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Játékos mozgatása
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
        player_x += player_speed
    if keys[pygame.K_SPACE]:
        bullets.append([player_x + player_size // 2 - bullet_size // 2, player_y])

    # Lövedékek mozgatása
    for bullet in bullets:
        bullet[1] -= bullet_speed
    bullets = [bullet for bullet in bullets if bullet[1] > 0]

    # Új ellenségek generálása
    if random.randint(1, 50) == 1:
        enemies.append([random.randint(0, WIDTH - enemy_size), 0])

    # Ellenségek mozgatása
    for enemy in enemies:
        enemy[1] += enemy_speed

    # Lövedékek és ellenségek ütközése
    for bullet in bullets:
        for enemy in enemies:
            if (enemy[0] < bullet[0] < enemy[0] + enemy_size and
                    enemy[1] < bullet[1] < enemy[1] + enemy_size):
                bullets.remove(bullet)
                enemies.remove(enemy)
                score += 1
                break

    # Játékos és ellenség ütközése
    for enemy in enemies:
        if enemy[1] + enemy_size > player_y and enemy[0] < player_x + player_size and enemy[0] + enemy_size > player_x:
            running = False

    # Rajzolás
    draw_player(player_x, player_y)
    draw_bullets()
    draw_enemies()

    # Pontszám kiírása
    score_text = font.render(f"Pontszám: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)  # FPS beállítás

pygame.quit()
