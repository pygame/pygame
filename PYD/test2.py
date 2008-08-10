# The python file is under test

import pygame
import physics
from pygame.locals import *

def render_body(body,surface,color):
    l = body.get_points()
    pygame.draw.polygon(surface,color,l)

def render_joint(joint,surface,color):
	l = joint.get_points()
	pygame.draw.lines(surface,color,False,l)

def render_world(world,surface,body_color,joint_color):
    for body in world.body_list:
        render_body(body,surface,body_color)
    for joint in world.joint_list:
        render_joint(joint,surface,joint_color)        

def init_world():
    w = physics.World()
    w.gravity = 0, 5
    
    body = physics.Body()
    body.shape = physics.RectShape (20, 20, 0)
    body.position = 200, 100
    body.restitution = 1.0
    body.static = True
    w.add_body(body)
    body1 = physics.Body()
    body1.shape = physics.RectShape (20,20,0)
    body1.position = 200, 200
    body1.restitution = 1.0
    w.add_body(body1)
    body2 = physics.Body()
    body2.shape = physics.RectShape (20,20,0)
    body2.position = 300, 200
    body1.restitution = 1.0
    w.add_body(body2)
    
    joint1 = physics.DistanceJoint(body1,body,1)
    joint1.anchor1 = 0, 0
    joint1.anchor2 = 0, 0
    w.add_joint(joint1)
    joint2 = physics.DistanceJoint(body1,body2,1)
    joint2.anchor1 = 0, 0
    joint2.anchor2 = 0, 0
    w.add_joint(joint2)
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
        
    white = 250, 250, 250
    red = 255,0,0
    world = init_world()
    
    
#Main Loop
    while 1:
        t = clock.tick(60)
        t = t/1000.0;
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
        render_world(world,background,white,red)
        screen.blit(background, (0, 0))
        pygame.display.flip()
        

#Game Over


#this calls the 'main' function when this script is executed
if __name__ == '__main__': main()
