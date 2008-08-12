# The python file is under test

import pygame
import physics
from pygame.locals import *

def render_body(body,surface,color):
    l = body.get_points()
    pygame.draw.polygon(surface,color,l)

def render_world(world,surface,color):
    for body in world.bodies:
        render_body(body,surface,color)
        

def init_world():
    w = physics.World()
    w.gravity = 0, 1
    
    for i in range(1, 30):
        body1 = physics.Body()
        body1.shape = physics.RectShape(30, 28, 0)
        body1.position = 400, -900 + 40*i
        body1.rotation = i
        body1.restitution = 0.0
        body1.mass = 20
        w.add_body(body1)
    
    
    body2 = physics.Body()
    body2.shape = physics.RectShape (760, 20, 0)
    body2.position = 400, 600
    body2.restitution = 0.0
    body2.mass = 1e100
    body2.static = True
    body2.rotation = 0.2
    w.add_body(body2)
    
    return w


def main():
    """this function is called when the program starts.
       it initializes everything it needs, then runs in
       a loop until the function returns."""
#Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption('physics test')

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
    white = 0, 250, 250
    theta = 0
    world = init_world()
    
    
    
#Main Loop
    while 1:
        # t = clock.tick(60)
        # t = t/1000.0;
        world.update(0.1)
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
        #print world.body_list[0]
        render_world(world,background,white)
        screen.blit(background, (0, 0))
        pygame.display.flip()
        

#Game Over


#this calls the 'main' function when this script is executed
if __name__ == '__main__': main()


