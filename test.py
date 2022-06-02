import pygame
import time

start = time.time()

screen = pygame.display.set_mode((500, 500))

for e in pygame.event.get():
    if e.type == pygame.QUIT:
        pygame.quit()

screen.blit(screen, (0, 0))

end = time.time()

print(f"This program took {end - start}s")