#/usr/bin/env python

"""This simple example is used for the line-by-line tutorial
that comes with pygame. It is based on a 'popular' web banner.
Note there are comments here, but for the full explanation, 
follow along in the tutorial. This code contains little error
checking to make it a little clearer. The full tutorial explains
where and how better error checking will help."""


#Import Modoules
import os
import pygame, pygame.font, pygame.image, pygame.mixer
from pygame.locals import *


#Resource Filenames
chimpfile = os.path.join('data', 'chimp.gif')
fistfile = os.path.join('data', 'fist.gif')
hitfile = os.path.join('data', 'punch.wav')
missfile = os.path.join('data', 'whiff.wav')


def main():
#Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((468, 60), HWSURFACE|DOUBLEBUF)
    pygame.display.set_caption('Monkey Fever')
    pygame.mouse.set_visible(0)
     
#Create The Backgound
    background = pygame.Surface(screen.get_size())
    background.fill((250, 250, 250))
    
#Put Text On The Background, Centered
    font = pygame.font.Font(None, 36)
    text = font.render("Pummel The Chimp, And Win $$$", 1, (10, 10, 10))
    textpos = text.get_rect()
    textpos.centerx = background.get_rect().centerx
    background.blit(text, textpos)

#Display The Background While Setup Finishes
    screen.blit(background, (0, 0))
    pygame.display.flip()
    
#Load Resources
    chimp = pygame.image.load(chimpfile).convert()
    fist = pygame.image.load(fistfile).convert()
    whiffsound = pygame.mixer.Sound(missfile)
    hitsound = pygame.mixer.Sound(hitfile)
    
#Prepare To Animate
    chimppos = chimp.get_rect()
    chimppos.bottom = screen.get_height()
    chimpmove = 2
    reload = 0

#Main Loop
    while 1:
    
    #Handle Input, Check For Quit
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
    
    #Move The Monkey
        chimppos.left += chimpmove
        if not screen.get_rect().contains(chimppos):
            chimpmove = -chimpmove
    
    #Move And Punch The Fist
        fistpos = pygame.mouse.get_pos()    
        pressed = pygame.mouse.get_pressed()[0]
        if not reload and pressed:
            if chimppos.collidepoint(fistpos):
                hitsound.play()
            else:
                whiffsound.play()
        reload = pressed
        if not reload:
            fistpos = fistpos[0] - 20, fistpos[1] - 10

    #Draw The Entire Scene
        screen.blit(background, (0, 0))
        screen.blit(chimp, chimppos)
        screen.blit(fist, fistpos)
        pygame.display.flip()

#Game Over


#this is python code to kickstart the program if not imported
if __name__ == '__main__': main()
