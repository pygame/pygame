import pygame

def get_delta_time(clock, fps):
    return clock.tick(fps) / 1000

# Sample usage. The value of FPS can be modified as per requirement.
FPS = 60
clock = pygame.time.Clock()

while True:
    delta = get_delta_time(clock, FPS)
    # Using delta for frame-independent calculations.
