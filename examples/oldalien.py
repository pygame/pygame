#! /usr/bin/env python

#import
import whrandom, os.path, sys
import pygame, pygame.image
from pygame.locals import *


#constants
FRAMES_PER_SEC = 50
PLAYER_SPEED   = 5
MAX_SHOTS      = 2
SHOT_SPEED     = 6
ALIEN_SPEED    = 6
ALIEN_ODDS     = 45
EXPLODE_TIME   = 6
SCREENRECT     = rect(0, 0, 640, 480)


#some globals for friendly access
dirtyrects = [] # list of update_rects
next_tick = 0   # used for timing
class Img: pass # container for images



#first, we define some utility functions
    
def load_image(file, transparent):
    "loads an image, prepares it for play"
    file = os.path.join('data', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit, 'Could not load image "%s" %s'%(file, pygame.get_error())
    if transparent:
        corner = surface.get_at((0, 0))
        surface.set_colorkey(corner, RLEACCEL)
    return surface.convert()



def wait_frame():
    "wait for the correct fps time to expire"
    global next_tick
    this_tick = pygame.time.get_ticks()
    if this_tick < next_tick:
        pygame.time.delay(next_tick - this_tick)
    next_tick = this_tick + (1000/FRAMES_PER_SEC)



# The logic for all the different sprite types

class Actor:
    "An enhanced sort of sprite class"
    def __init__(self, image):
        self.image = image
        self.rect = rect((0, 0), image.get_size())
        
    def update(self):
        "update the sprite state for this frame"
        pass
    
    def draw(self, screen):
        "draws the sprite into the screen"
        r = screen.blit(self.image, self.rect.topleft)
        dirtyrects.append(r)
        
    def erase(self, screen, background):
        "gets the sprite off of the screen"
        pos = self.rect.topleft
        r = screen.blit(background, pos, pos, self.rect.size)
        dirtyrects.append(r)


class Player(Actor):
    "Cheer for our hero"
    def __init__(self):
        Actor.__init__(self, Img.player)
        self.alive = 1
        self.reloading = 0
        self.rect.centerx = SCREENRECT.centerx
        self.rect.bottom = SCREENRECT.bottom - 10

    def move(self, direction):
        self.rect = self.rect.move(direction*PLAYER_SPEED, 0).clamp(SCREENRECT)


class Alien(Actor):
    "Destroy him or suffer"
    def __init__(self):
        Actor.__init__(self, Img.alien)
        self.facing = whrandom.choice((-1,1)) * ALIEN_SPEED
        if self.facing < 0:
            self.rect.right = SCREENRECT.right
            
    def update(self):
        global SCREENRECT
        self.rect[0] += self.facing
        if not SCREENRECT.contains(self.rect):
            self.facing *= -1;
            self.rect.top = self.rect.bottom + 3
            self.rect = self.rect.clamp(SCREENRECT)


class Explosion(Actor):
    "Beware the fury"
    def __init__(self, actor):
        Actor.__init__(self, Img.explosion)
        self.life = EXPLODE_TIME
        self.rect.center = actor.rect.center
        
    def update(self):
        self.life -= 1


class Shot(Actor):
    "The big payload"
    def __init__(self, player):
        Actor.__init__(self, Img.shot)
        self.rect.centerx = player.rect.centerx
        self.rect.top = player.rect.top - 10

    def update(self):
        self.rect.top -= SHOT_SPEED
        



def main():
    "Run me for adrenaline"
    global dirtyrects

    # Initialize SDL components
    pygame.init()
    screen = pygame.display.set_mode(SCREENRECT.size, 0)

    # Load the Resources
    Img.background = load_image('background.gif', 0)
    Img.shot = load_image('shot.gif', 1)
    Img.bomb = load_image('bomb.gif', 1)
    Img.danger = load_image('danger.gif', 1)
    Img.alien = load_image('alien1.gif', 1)
    Img.player = load_image('oldplayer.gif', 1)
    Img.explosion = load_image('explosion1.gif', 1)

    # Create the background
    background = pygame.surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, Img.background.get_size()[0]):
        background.blit(Img.background, (x, 0))
    screen.blit(background, (0,0))
    pygame.display.flip()

    # Initialize Game Actors
    player = Player()
    aliens = [Alien()]
    shots = []
    explosions = []

    # Main loop
    while player.alive or explosions:
        wait_frame()

        # Gather Events
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        if keystate[K_ESCAPE] or pygame.event.peek(QUIT):
            break

        # Clear screen and update actors
        for actor in [player] + aliens + shots + explosions:
            actor.erase(screen, background)
            actor.update()
        
        # Clean Dead Explosions and Bullets
        for e in explosions:
            if e.life <= 0:
                explosions.remove(e)
        for s in shots:
            if s.rect.top <= 0:
                shots.remove(s)

        # Move the player
        direction = keystate[K_RIGHT] - keystate[K_LEFT]
        player.move(direction)

        # Create new shots
        if not player.reloading and keystate[K_SPACE] and len(shots) < MAX_SHOTS:
            shots.append(Shot(player))
        player.reloading = keystate[K_SPACE]

        # Create new alien
        if not int(whrandom.random() * ALIEN_ODDS):
            aliens.append(Alien())

        # Detect collisions
        alienrects = [a.rect for a in aliens]
        hit = player.rect.collidelist(alienrects)
        if hit != -1:
            alien = aliens[hit]
            explosions.append(Explosion(alien))
            explosions.append(Explosion(player))
            aliens.remove(alien)
            player.alive = 0
        for shot in shots:
            hit = shot.rect.collidelist(alienrects)
            if hit != -1:
                alien = aliens[hit]
                explosions.append(Explosion(alien))
                shots.remove(shot)
                aliens.remove(alien)
                break

        # Draw everybody
        for actor in [player] + aliens + shots + explosions:
            actor.draw(screen)

        pygame.display.update(dirtyrects)
        dirtyrects = []

    pygame.time.delay(100)
    

#if python says run, let's run!
if __name__ == '__main__':
    main()
    