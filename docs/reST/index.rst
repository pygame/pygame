import pygame
import sys

# Inicializar Pygame
pygame.init()

# Definir constantes
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Juego de Plataforma Simple")

# Variables del personaje
player_size = 50
player_x = WIDTH // 2
player_y = HEIGHT - player_size
player_velocity = 5

# Bucle del juego
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_velocity
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
        player_x += player_velocity
    if keys[pygame.K_UP] and player_y > 0:
        player_y -= player_velocity
    if keys[pygame.K_DOWN] and player_y < HEIGHT - player_size:
        player_y += player_velocity

    screen.fill(WHITE)  # Limpiar la pantalla
    pygame.draw.rect(screen, RED, (player_x, player_y, player_size, player_size))  # Dibujar al personaje
    pygame.display.flip()  # Actualizar la pantalla

    pygame.time.Clock().tick(30)  # Regular la velocidad del juego
