#! /usr/bin/env python

"""This is a pretty full-fledged example of a miniature
game with pygame. It's not what you'd call commercial grade
quality, but it does demonstrate just about all of the
important modules for pygame. Note the methods it uses to
detect the availability of the font and mixer modules.
This example actually gets a bit large. A better starting
point for beginners would be the oldaliens.py example."""


import whrandom, os.path, sys
import pygame, pygame.image
from pygame.locals import *
try:
    import pygame.mixer
    pygame.mixer.pre_init(11025)
except:
    pygame.mixer = None

#see if we can get some font lovin'
try: import pygame.font as font
except ImportError: font = None


#constants
FRAMES_PER_SEC = 45
PLAYER_SPEED   = 6
MAX_SHOTS      = 2
SHOT_SPEED     = 9
BOMB_SPEED     = 9
MAX_ALIENS     = 30
ALIEN_SPEED    = 9
ALIEN_ODDS     = 29
ALIEN_RELOAD   = 12
EXPLODE_TIME   = 35
MAX_EXPLOSIONS = 4
SCREENRECT     = Rect(0, 0, 640, 480)
ANIMCYCLE      = 12
PLODECYCLE     = 7
BULLET_OFFSET  = 11
BOUNCEWIDTH    = PLAYER_SPEED * 4
DIFFICULTY     = 16
BOMB_ODDS      = 130
DANGER         = 10
SORRYSCORE     = 20
GOODSCORE      = 60

#some globals for friendly access
dirtyrects = [] # list of update_rects
class Img: pass # container for images
class Snd: pass # container for sounds




#first, we define some utility functions

class dummysound:
    def play(self): pass

    
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



def load_sound(file):
    if not pygame.mixer: return dummysound()
    file = os.path.join('data', file)
    try:
        sound = pygame.mixer.Sound(file)
        return sound
    except pygame.error:
        print 'Warning, unable to load,', file
    return dummysound()
    


last_tick = 0
ticks_per_frame = int((1.0 / FRAMES_PER_SEC) * 1000.0)
def wait_frame():
    "wait for the correct fps time to expire"
    global last_tick, ticks_per_frame
    now = pygame.time.get_ticks()
    wait = ticks_per_frame - (now - last_tick)
    pygame.time.delay(wait)
    last_tick = pygame.time.get_ticks()



# The logic for all the different sprite types

class Actor:
    "An enhanced sort of sprite class"
    def __init__(self, image):
        self.image = image
        self.rect = image.get_rect()
        self.clearrect = self.rect
        
    def update(self):
        "update the sprite state for this frame"
        pass
    
    def draw(self, screen):
        "draws the sprite into the screen"
        r = screen.blit(self.image, self.rect)
        dirtyrects.append(r.union(self.clearrect))
        
    def erase(self, screen, background):
        "gets the sprite off of the screen"
        r = screen.blit(background, self.rect, self.rect)
        self.clearrect = r


class Player(Actor):
    "Cheer for our hero"
    def __init__(self):
        Actor.__init__(self, Img.player[0])
        self.alive = 1
        self.reloading = 0
        self.rect.centerx = SCREENRECT.centerx
        self.rect.bottom = SCREENRECT.bottom - 1
        self.origtop = self.rect.top

    def move(self, direction):
        self.rect = self.rect.move(direction*PLAYER_SPEED, 0).clamp(SCREENRECT)
        if direction < 0:
            self.image = Img.player[0]
        elif direction > 0:
            self.image = Img.player[1]
        self.rect.top = self.origtop - (self.rect.left/BOUNCEWIDTH%2)


class Alien(Actor):
    "Destroy him or suffer"
    def __init__(self):
        Actor.__init__(self, Img.alien[0])
        self.facing = whrandom.choice((-1,1)) * ALIEN_SPEED
        self.frame = 0
        if self.facing < 0:
            self.rect.right = SCREENRECT.right
            
    def update(self):
        global SCREENRECT
        self.rect[0] += self.facing
        if not SCREENRECT.contains(self.rect):
            self.facing *= -1;
            self.rect.top = self.rect.bottom + 1
            self.rect = self.rect.clamp(SCREENRECT)
        self.frame += 1
        self.image = Img.alien[self.frame/ANIMCYCLE%3]


class Explosion(Actor):
    "Beware the fury"
    def __init__(self, actor, longer=0):
        Snd.explosion.play()
        Actor.__init__(self, Img.explosion[0])
        self.life = EXPLODE_TIME
        self.rect.center = actor.rect.center
        if longer:
            self.life *= 2
        
    def update(self):
        self.life -= 1
        self.image = Img.explosion[self.life/PLODECYCLE%2]


class Shot(Actor):
    "The big payload"
    def __init__(self, player):
        Snd.shot.play()
        Actor.__init__(self, Img.shot)
        self.rect.centerx = player.rect.centerx
        self.rect.top = player.rect.top - 10
        if player.image is Img.player[0]:
            self.rect.left += BULLET_OFFSET
        elif player.image is Img.player[1]:
            self.rect.left -= BULLET_OFFSET

    def update(self):
        self.rect = self.rect.move(0, -SHOT_SPEED)
        
class Bomb(Actor):
    "The big payload"
    def __init__(self, alien):
        Actor.__init__(self, Img.bomb)
        self.rect.centerx = alien.rect.centerx
        self.rect.bottom = alien.rect.bottom + 5

    def update(self):
        self.rect = self.rect.move(0, BOMB_SPEED)

class Danger(Actor):
    "Here comes trouble"
    def __init__(self):
        Actor.__init__(self, Img.danger)
        self.life = 1
        self.tick = 0
        self.rect.center = SCREENRECT.center[0]-30, 30
        self.startleft = self.rect.left
        
    def update(self):
        self.tick += 1
        self.rect.left = self.startleft + (self.tick/25%2)*60


def main(winstyle = 0):
    "Run me for adrenaline"
    global dirtyrects

    # Initialize SDL components
    pygame.init()
    screen = pygame.display.set_mode(SCREENRECT.size, winstyle)

    if pygame.joystick.get_init() and pygame.joystick.get_count():
        joy = pygame.joystick.Joystick(0)
        joy.init()
        if not joy.get_numaxes() or not joy.get_numbuttons():
            print 'warning: joystick disabled. requires at least one axis and one button'
            joy.quit()
            joy = None
    else:
        joy = None

    #check that audio actually initialized
    if pygame.mixer and not pygame.mixer.get_init():
        pygame.mixer = None
    if not pygame.mixer:
        print 'Warning, sound disabled'


    # Load the Resources
    Img.background = load_image('background.gif', 0)
    Img.shot = load_image('shot.gif', 1)
    Img.bomb = load_image('bomb.gif', 1)
    Img.danger = load_image('danger.gif', 1)
    Img.alien = load_image('alien1.gif', 1), \
                load_image('alien2.gif', 1), \
                load_image('alien3.gif', 1)
    Img.player = load_image('player1.gif', 1), \
                 load_image('player2.gif', 1)
    Img.explosion = load_image('explosion1.gif', 1), \
                    load_image('explosion2.gif', 1)
    Img.explosion[0].set_alpha(128, RLEACCEL)
    Img.explosion[1].set_alpha(128, RLEACCEL)
    Img.danger.set_alpha(128, RLEACCEL)

    Snd.explosion = load_sound('boom.wav')
    Snd.shot = load_sound('car_door.wav')

    # Create the background
    background = pygame.Surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, Img.background.get_width()):
        background.blit(Img.background, (x, 0))
    screen.blit(background, (0,0))
    pygame.display.flip()
    pygame.mouse.set_visible(0)

    # Initialize Game Actors
    player = Player()
    aliens = [Alien()]
    shots = []
    bombs = []
    explosions = []
    misc = []
    alienreload = ALIEN_RELOAD
    difficulty = DIFFICULTY
    bomb_odds = BOMB_ODDS
    kills = 0

    # Soundtrack
    if pygame.mixer:
        music = os.path.join('data', 'house_lo.wav')
        pygame.mixer.music.load(music)
        pygame.mixer.music.play(-1)

    # Main loop
    while player.alive or explosions:
        wait_frame()

        # Gather Events
        pygame.event.pump()
        keystate = pygame.key.get_pressed()
        if keystate[K_ESCAPE] or pygame.event.peek(QUIT):
            break
        

        if difficulty:
            difficulty -= 1
        else:
            difficulty = DIFFICULTY
            if bomb_odds > DANGER:
                bomb_odds -= 1
                if bomb_odds == DANGER:
                    misc.append(Danger())

        # Clear screen and update actors
        for actor in [player] + aliens + shots + bombs + explosions + misc:
            actor.erase(screen, background)
            actor.update()
        
        # Clean Dead Explosions and Bullets
        for e in explosions:
            if e.life <= 0:
                dirtyrects.append(e.clearrect)
                explosions.remove(e)
        for s in shots:
            if s.rect.top <= 0:
                dirtyrects.append(s.clearrect)
                shots.remove(s)
        for b in bombs:
            if b.rect.bottom >= 470:
                if player.alive:
                    explosions.append(Explosion(b))
                dirtyrects.append(b.clearrect)
                bombs.remove(b)

        # Handle Input, Control Tank
        if player.alive:
            direction = keystate[K_RIGHT] - keystate[K_LEFT]
            firing = keystate[K_SPACE]
            if joy:
                direction += joy.get_axis(0)
                firing += joy.get_button(0)
            player.move(direction)
            if not player.reloading and firing and len(shots) < MAX_SHOTS:
                shots.append(Shot(player))
            player.reloading = firing

        # Create new alien
        if alienreload:
            alienreload -= 1
        elif player.alive and not int(whrandom.random() * ALIEN_ODDS):
            aliens.append(Alien())
            alienreload = ALIEN_RELOAD

        # Drop bombs
        if player.alive and aliens and not int(whrandom.random() * bomb_odds):
            bombs.append(Bomb(aliens[-1]))

        # Detect collisions
        alienrects = [a.rect for a in aliens]
        hit = player.rect.collidelist(alienrects)
        if hit != -1:
            alien = aliens[hit]
            explosions.append(Explosion(alien))
            explosions.append(Explosion(player, 1))
            dirtyrects.append(alien.clearrect)
            aliens.remove(alien)
            kills += 1
            player.alive = 0
        for shot in shots:
            hit = shot.rect.collidelist(alienrects)
            if hit != -1:
                alien = aliens[hit]
                explosions.append(Explosion(alien))
                dirtyrects.append(shot.clearrect)
                dirtyrects.append(alien.clearrect)
                shots.remove(shot)
                aliens.remove(alien)
                kills += 1
                break
        bombrects = [b.rect for b in bombs]
        hit = player.rect.collidelist(bombrects)
        if hit != -1:
            bomb = bombs[hit]
            explosions.append(Explosion(bomb))
            explosions.append(Explosion(player, 1))
            dirtyrects.append(bomb.clearrect)
            bombs.remove(bomb)
            player.alive = 0

        #prune the explosion list
        diff = len(explosions) - MAX_EXPLOSIONS
        if diff > 0:
            for x in range(diff):
                dirtyrects.append(explosions[x].clearrect)
            explosions = explosions[diff:]

        # Draw everybody
        for actor in [player] + bombs + aliens + shots + explosions + misc:
            actor.draw(screen)

        pygame.display.update(dirtyrects)
        dirtyrects = []

    if pygame.mixer:
        pygame.mixer.music.fadeout(1200)

    #attempt to show game over (if font installed)
    if font:
        f = font.Font(None, 100) #None means default font
        f.set_italic(1)
        text = f.render('Game Over', 1, (200, 200, 200))
        textrect = Rect((0, 0), text.get_size())
        textrect.center = SCREENRECT.center
        screen.blit(text, textrect)
        pygame.display.flip()

    #wait a beat
    if pygame.mixer:
        while pygame.mixer.music.get_busy():
            pygame.time.delay(200)
    else:
        pygame.time.delay(800)


    #scoreboard
    print 'Total Kills =', kills
    if kills <= SORRYSCORE: print 'Sorry!'
    elif kills >= GOODSCORE: print 'Excellent!'
    elif kills >= GOODSCORE-12: print 'Almost Excellent!'



    

#if python says run, let's run!
if __name__ == '__main__':
    main()
    
