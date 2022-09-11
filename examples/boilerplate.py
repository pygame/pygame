import pygame

pygame.init()
# define variables
screen = pygame.display.set_mode((1280, 720))

clock = pygame.time.Clock()


# main loop
while True:
    # fill the screen with the color purple
    screen.fill("purple")
# get pygame events
    for event in pygame.event.get():
        # if pygame is quitted, end the program
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
# update the screen
    pygame.display.flip()

# make the game go at 60 fps
    clock.tick(60)
