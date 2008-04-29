#!/usr/bin/env python
import sys, random
import pygame, pygame.image, pygame.surface, pygame.time, pygame.display

def maskFromSurface(surface, threshold = 127):
    #return pygame.mask.from_surface(surface, threshold)

    mask = pygame.mask.Mask(surface.get_size())
    key = surface.get_colorkey()
    if key:
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                if surface.get_at((x+0.1,y+0.1)) != key:
                    mask.set_at((x,y),1)
    else:
        for y in range(surface.get_height()):
            for x in range (surface.get_width()):
                if surface.get_at((x,y))[3] > threshold:
                    mask.set_at((x,y),1)
    return mask

def vadd(x,y):
    return [x[0]+y[0],x[1]+y[1]]

def vsub(x,y):
    return [x[0]-y[0],x[1]-y[1]]

def vdot(x,y):
    return x[0]*y[0]+x[1]*y[1]

class Sprite:
    def __init__(self, surface, mask = None):
        self.surface = surface
        if mask:
            self.mask = mask
        else:
            self.mask = maskFromSurface(self.surface)
        self.setPos([0,0])
        self.setVelocity([0,0])
        
    def setPos(self,pos):
        self.pos = [pos[0],pos[1]]
    def setVelocity(self,vel):
        self.vel = [vel[0],vel[1]]
    def move(self,dr):
        self.pos = vadd(self.pos,dr)
    def kick(self,impulse):
        self.vel[0] += impulse[0]
        self.vel[1] += impulse[1]

    def collide(self,s):
        """Test if the sprites are colliding and
        resolve the collision in this case."""
        offset = map(int,vsub(s.pos,self.pos))
        overlap = self.mask.overlap_area(s.mask,offset)
        if overlap == 0:
            return
        """Calculate collision normal"""
        nx = (self.mask.overlap_area(s.mask,(offset[0]+1,offset[1])) -
              self.mask.overlap_area(s.mask,(offset[0]-1,offset[1])))
        ny = (self.mask.overlap_area(s.mask,(offset[0],offset[1]+1)) -
              self.mask.overlap_area(s.mask,(offset[0],offset[1]-1)))
        if nx == 0 and ny == 0:
            """One sprite is inside another"""
            return
        n = [nx,ny]
        dv = vsub(s.vel,self.vel)
        J = vdot(dv,n)/(2*vdot(n,n))
        if J > 0:
            """Can scale up to 2*J here to get bouncy collisions"""
            J *= 1.9
            self.kick([nx*J,ny*J])
            s.kick([-J*nx,-J*ny])
        return
        """Separate the sprites"""
        c1 = -overlap/vdot(n,n)
        c2 = -c1/2
        self.move([c2*nx,c2*ny])
        s.move([(c1+c2)*nx,(c1+c2)*ny])
        
    def update(self,dt):
        self.pos[0] += dt*self.vel[0]
        self.pos[1] += dt*self.vel[1]


def main(argv):
    if len(argv) < 2:
        print 'Usage: pygame_example IMAGE'
        print 'Let many copies of IMAGE bounce against each other'
        print 'Press any key to quit'
        return
    print 'Press any key to quit'
    screen = pygame.display.set_mode((640,480))
    images = []
    masks = []
    for impath in argv[1:]:
        images.append(pygame.image.load(impath).convert_alpha())
        masks.append(maskFromSurface(images[-1]))

    numtimes = 10
    import time
    t1 = time.time()
    for x in range(numtimes):
        m = maskFromSurface(images[-1])
    t2 = time.time()

    print "python maskFromSurface :%s" % (t2-t1)

    t1 = time.time()
    for x in range(numtimes):
        m = pygame.mask.from_surface(images[-1])
    t2 = time.time()

    print "C pygame.mask.from_surface :%s" % (t2-t1)

    sprites = []
    for i in range(20):
        j = i % len(images)
        s = Sprite(images[j],masks[j])
        s.setPos((random.uniform(0,screen.get_width()),
                  random.uniform(0,screen.get_height())))
        s.setVelocity((random.uniform(-5,5),random.uniform(-5,5)))
        sprites.append(s)
    pygame.time.set_timer(pygame.USEREVENT,33)
    while 1:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return
        elif event.type == pygame.USEREVENT:
            """Do both mechanics and screen update"""
            screen.fill((240,220,100))
            for i in range(len(sprites)):
                for j in range(i+1,len(sprites)):
                    sprites[i].collide(sprites[j])
            for s in sprites:
                s.update(1)
                if s.pos[0] < -s.surface.get_width()-3:
                    s.pos[0] = screen.get_width()
                elif s.pos[0] > screen.get_width()+3:
                    s.pos[0] = -s.surface.get_width()
                if s.pos[1] < -s.surface.get_height()-3:
                    s.pos[1] = screen.get_height()
                elif s.pos[1] > screen.get_height()+3:
                    s.pos[1] = -s.surface.get_height()
                screen.blit(s.surface,s.pos)
            pygame.display.update()
        elif event.type == pygame.KEYDOWN:
            return

main(sys.argv)



        
