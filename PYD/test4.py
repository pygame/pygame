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
    body.shape = physics.RectShape(1, 1, 0)
    body.position = 400, 20
    body.restitution = 0.0
    body.static = True
    w.add_body(body)
    pre_body = body
    for i in range(1, 30):
        body = physics.Body()
        body.shape = physics.RectShape(5, 5, 0)
        body.position = 400+10*i, 10*i+20
        body.restitution = 0.0
        body.mass = 20
        w.add_body(body)
        joint = physics.DistanceJoint(pre_body,body,0)
        joint.anchor1 = 0, 0
        joint.anchor2 = 0, 0
        w.add_joint(joint)
        pre_body = body
        
    
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
