# pygame setup
import pygame

pygame.init()
screen = pygame.display.set_mode([600, 400])
pygame.display.set_caption("Example code for the cursors module")

# create a system cursor
system = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_NO)

# create bitmap cursors
bitmap_1 = pygame.cursors.Cursor(*pygame.cursors.arrow)
bitmap_2 = pygame.cursors.Cursor(
    (24, 24), (0, 0), *pygame.cursors.compile(pygame.cursors.thickarrow_strings)
)

# create a color cursor
surf = pygame.Surface((40, 40)) # you could also load an image 
surf.fill((120, 50, 50))        # and use that as your surface
color = pygame.cursors.Cursor((20, 20), surf)

cursors = [system, bitmap_1, bitmap_2, color]
cursor_index = 0

pygame.mouse.set_cursor(cursors[cursor_index])

clock = pygame.time.Clock()
while True:
    clock.tick(60)
    screen.fill((0, 75, 30))
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        # if the mouse is clicked it will switch to a new cursor
        if event.type == pygame.MOUSEBUTTONDOWN:
            cursor_index += 1
            cursor_index %= len(cursors)
            pygame.mouse.set_cursor(cursors[cursor_index])
