# The python file is under test

import pygame
import physics
from pygame.locals import *


def render_body(body,surface,color):
    l = physics.get_point_list(body)
    pygame.draw.polygon(surface,color,l)


def render_world(world,surface,color):
    for body in world.body_list:
        render_body(body,color)
        

def init_world():
    w = physics.World()
    body1 = physics.Body()
    body1.bind_rect_shape(2,2,0)
    body1.position = 1+2j
    w.add_body(body1)
    return w


def main():
    """this function is called when the program starts.
       it initializes everything it needs, then runs in
       a loop until the function returns."""
#Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption('physics test')
    pygame.mouse.set_visible(0)

#Create The Backgound
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0,0,0))


#Display The Background
    screen.blit(background, (0, 0))
    pygame.display.flip()

#Prepare Game Objects
    clock = pygame.time.Clock()
        
    #rect = pygame.Rect(0,0,20,20)
    #pointlist = [(0,10),(10,0),(0,0)]
    white = 250, 250, 250
    theta = 0
    
    world = init_world()
    
#Main Loop
    while 1:
        t = clock.tick(60)
        t = t/1000.0;
        world.update(t)
    #Handle Input Events
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            elif event.type == MOUSEBUTTONDOWN:
                return
            elif event.type is MOUSEBUTTONUP:
                return
        

    #Draw Everything
        background.fill((0,0,0))
        render_world(world,background,white)
        screen.blit(background, (0, 0))
        pygame.display.flip()

#Game Over


#this calls the 'main' function when this script is executed
if __name__ == '__main__': main()


