#! /usr/bin/env python

import random, os.path, sys

#import basic pygame modules
import pygame, pygame.image, pygame.transform, pygame.sprite
from pygame.locals import *

#see if we can load more than standard BMP
if not pygame.image.get_extended():
    raise SystemExit, "Sorry, extended image module required"


#try importing pygame optional modules
try:
    import pygame.font as font
except ImportError:
    print 'Warning, no fonts'
    font = None
try:
    import pygame.mixer
except ImportError:
    print 'Warning, no sound'
    pygame.mixer = None


#game constants
MAX_SHOTS      = 2      #most player bullets onscreen
ALIEN_ODDS     = 29     #chances a new alien appears
ALIEN_RELOAD   = 12     #frames between new aliens
BOMB_ODDS      = 200
SCREENRECT     = Rect(0, 0, 640, 480)


    
def load_image(file):
    "loads an image, prepares it for play"
    file = os.path.join('data', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit, 'Could not load image "%s" %s'%(file, pygame.get_error())
    return surface.convert()

def load_images(*files):
    return [load_image(file) for file in files]


class dummysound:
    def play(self): pass
    
def load_sound(file):
    if not pygame.mixer: return dummysound()
    file = os.path.join('data', file)
    try:
        sound = pygame.mixer.Sound(file)
        return sound
    except pygame.error:
        print 'Warning, unable to load,', file
    return dummysound()




class Player(pygame.sprite.Sprite):
    speed = 11
    bounce = 24
    gun_offset = -11
    images = []
    def __init__(self):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.reloading = 0
        self.rect.centerx = SCREENRECT.centerx
        self.rect.bottom = SCREENRECT.bottom - 1
        self.origtop = self.rect.top
        self.facing = -1

    def update(self): pass

    def move(self, direction):
        if direction: self.facing = direction
        self.rect.move_ip(direction*self.speed, 0)
        self.rect = self.rect.clamp(SCREENRECT)
        if direction < 0:
            self.image = self.images[0]
        elif direction > 0:
            self.image = self.images[1]
        self.rect.top = self.origtop - (self.rect.left/self.bounce%2)

    def gunpos(self):
        pos = self.facing*self.gun_offset + self.rect.centerx
        return pos, self.rect.top


class Alien(pygame.sprite.Sprite):
    speed = 15 
    animcycle = 12
    images = []
    def __init__(self):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.facing = random.choice((-1,1)) * Alien.speed
        self.frame = 0
        if self.facing < 0:
            self.rect.right = SCREENRECT.right
            
    def update(self):
        self.rect.move_ip(self.facing, 0)
        if not SCREENRECT.contains(self.rect):
            self.facing = -self.facing;
            self.rect.top = self.rect.bottom + 1
            self.rect = self.rect.clamp(SCREENRECT)
        self.frame = self.frame + 1
        self.image = self.images[self.frame/self.animcycle%3]


class Explosion(pygame.sprite.Sprite):
    defaultlife = 10
    animcycle = 3
    images = []
    def __init__(self, actor, longer):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.life = self.defaultlife * (longer+1)
        self.rect.center = actor.rect.center
        
    def update(self):
        self.life = self.life - 1
        self.image = self.images[self.life/self.animcycle%2]
        if self.life <= 0: self.kill()


class Shot(pygame.sprite.Sprite):
    speed = -13
    images = []
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect.midbottom = pos

    def update(self):
        self.rect.move_ip(0, self.speed)
        if self.rect.top <= 0:
            self.kill()


class Bomb(pygame.sprite.Sprite):
    speed = 11
    images = []
    def __init__(self, alien):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect.centerx = alien.rect.centerx
        self.rect.bottom = alien.rect.bottom + 5

    def update(self):
        self.rect.move_ip(0, self.speed)
        if self.rect.bottom >= 470:
            Explosion(self, 0)
            self.kill()



def main(winstyle = 0):
    # Initialize pygame
    pygame.init()

    # Set the display mode
    winstyle = 0  #|FULLSCREEN
    bestdepth = pygame.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pygame.display.set_mode(SCREENRECT.size, winstyle, bestdepth)

    #Load images, assign to sprite classes
    #(do this before the classes are used, after screen setup)
    img = load_image('player1.gif')
    Player.images = [img, pygame.transform.flip(img, 1, 0)]
    img = load_image('explosion1.gif')
    Explosion.images = [img, pygame.transform.flip(img, 1, 1)]
    Alien.images = load_images('alien1.gif', 'alien2.gif', 'alien3.gif')
    Bomb.images = [load_image('bomb.gif')]
    Shot.images = [load_image('shot.gif')]

    #decorate the game window
    icon = pygame.transform.scale(Alien.images[0], (32, 32))
    pygame.display.set_icon(icon)
    pygame.display.set_caption('Pygame Aliens')
    pygame.mouse.set_visible(0)

    #create the background, tile the bgd image
    bgdtile = load_image('background.gif')
    background = pygame.Surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, bgdtile.get_width()):
        background.blit(bgdtile, (x, 0))
    screen.blit(background, (0,0))
    pygame.display.flip()

    #load the sound effects
    boom_sound = load_sound('boom.wav')
    shoot_sound = load_sound('car_door.wav')
    if pygame.mixer:
        music = os.path.join('data', 'house_lo.wav')
        pygame.mixer.music.load(music)
        pygame.mixer.music.play(-1)

    # Initialize Game Groups
    aliens = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    bombs = pygame.sprite.Group()
    all = pygame.sprite.RenderUpdates()
    lastalien = pygame.sprite.GroupSingle()

    #assign default groups to each sprite class
    Player.containers = all
    Alien.containers = aliens, all, lastalien
    Shot.containers = shots, all
    Bomb.containers = bombs, all
    Explosion.containers = all

    #Create Some Starting Values
    alienreload = ALIEN_RELOAD
    kills = 0
    clock = pygame.time.Clock()

    #initialize our starting sprites
    player = Player()
    Alien() #note, this 'lives' because it goes into a sprite group


    while player.alive():
        # update keyboard state
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        if keystate[K_ESCAPE] or pygame.event.peek(QUIT):
            break

        # clear/erase the last drawn sprites
        all.clear(screen, background)

        #update all the sprites
        all.update()
        
        #handle player input
        direction = keystate[K_RIGHT] - keystate[K_LEFT]
        player.move(direction)
        firing = keystate[K_SPACE]
        if not player.reloading and firing and len(shots) < MAX_SHOTS:
            Shot(player.gunpos())
            shoot_sound.play()
        player.reloading = firing

        # Create new alien
        if alienreload:
            alienreload -= 1
        elif not int(random.random() * ALIEN_ODDS):
            Alien()
            alienreload = ALIEN_RELOAD

        # Drop bombs
        if lastalien and not int(random.random() * BOMB_ODDS):
            Bomb(lastalien.sprite)

        # Detect collisions
        for a in pygame.sprite.spritecollide(player, aliens, 1):
            boom_sound.play()
            Explosion(a, 0)
            Explosion(player, 1)
            kills += 1
            player.kill()

        for alien in pygame.sprite.groupcollide(shots, aliens, 1, 1).keys():
            boom_sound.play()
            Explosion(alien, 0)
            kills += 1
                    
        for bomb in pygame.sprite.spritecollide(player, bombs, 1):         
            boom_sound.play()
            Explosion(bomb, 0)

        #draw the scene
        dirty = all.draw(screen)
        pygame.display.update(dirty)

        #cap the framerate
        clock.tick(30)

    pygame.mixer.music.fadeout(1000)
    pygame.time.delay(1000)

#no need to clean up, python automatically cleans up after us
   


#call the "main" function if running this script
if __name__ == '__main__': main()
    
