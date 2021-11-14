"""pygame.examples.nospritelayer

Shows a window where you can move a square... or can you ?

Pygame features showcased here :

- pg.sprite.NoSpriteLayer - a special singleton that you can use to make some sprites
'undrawable' (they won't get rendered on screen). Undrawable sprites can have multiple usages,
although the main one would be to make some conditional blocks in the code to prevent the player
from moving, for example during cutscenes. Here, it prevents the player's blue square from
moving while a red square crosses the screen.
-pg.sprite.LayeredUpdates - a group type in which you can control sprite rendering using
layers and NoSpriteLayer.

Controls :

Arrow Keys - move the blue square"""

import pygame

pygame.init()


class Square(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.pressed={pygame.K_UP:False,
                      pygame.K_DOWN:False,
                      pygame.K_LEFT:False,
                      pygame.K_RIGHT:False}
        self.image=pygame.Surface((64, 64))
        self.image.fill(pygame.Color("blue"))
        self.rect=self.image.get_rect(center=(300, 300))
    def update(self):
        if not mv_lock:
            if self.pressed.get(pygame.K_UP):
                self.rect.y-=4
            elif self.pressed.get(pygame.K_DOWN):
                self.rect.y+=4
            if self.pressed.get(pygame.K_LEFT):
                self.rect.x-=4
            elif self.pressed.get(pygame.K_RIGHT):
                self.rect.x+=4

class NPSquare(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image=pygame.Surface((64, 64))
        self.image.fill(pygame.Color("red"))
        self.rect=self.image.get_rect(center=(320, 320))
    def update(self):
        if self.rect.x>600:
            self.kill()
        self.rect.x+=4

class MovementLock(pygame.sprite.Sprite):
    pass

win=pygame.display.set_mode((600, 600))
pygame.display.set_caption("pygame.sprite.NoSpriteLayer showcase")
clock=pygame.time.Clock()
dt=0
duration=0
NSL=False
running=True
player=Square()
squares=pygame.sprite.LayeredUpdates(player)
mv_lock=pygame.sprite.LayeredUpdates()

while running:
    dt=clock.tick(60)
    duration+=dt
    if duration>=5000:
        NSL=not NSL
        if NSL:
            mv_lock.add(MovementLock(), layer=pygame.sprite.NoSpriteLayer)
            squares.add(NPSquare())
        else:
            mv_lock.empty()
        duration=0
    squares.update()
    for event in pygame.event.get([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT]):
        if event.type==pygame.QUIT:
            pygame.quit()
            running=False
            exit(0)
        elif event.type==pygame.KEYDOWN:
            if event.key in player.pressed:
                player.pressed[event.key]=True
        elif event.type==pygame.KEYUP:
            if event.key in player.pressed:
                player.pressed[event.key]=False
