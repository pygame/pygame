import pygame
pygame.init()
screen=pygame.display.set_mode((320,240), pygame.SCALED)
screen.fill((255,255,255))
pygame.display.flip()

z_buf = pygame.Surface((320,240), depth=8)
z_buf.fill(255)

sprit = pygame.Surface((20,20))
sprit.fill((255,0,255))
sprit.set_colorkey((255,0,255))
pygame.draw.circle(sprit, (100,100,100), (10,10), 8, 2)

screen.blit(sprit, (20,20))

pygame.draw.circle(sprit, (100,200,200), (10,10), 8, 2)
screen.depth_blit(sprit, (20,25), z_buf, 8)

pygame.draw.circle(sprit, (200,100,100), (10,10), 8, 2)
screen.depth_blit(sprit, (25,25), z_buf, 9)

pygame.display.flip()

input()
