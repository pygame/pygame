import sys
import pygame2
import pygame2.physics as physics

try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.image as image
    import pygame2.sdl.time as time
    import pygame2.sdl.wm as wm
    import pygame2.sdlext.draw as draw
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)

def render_body (screen, body):
    points = body.get_points ()
    draw.polygon (screen, white, points, 1)

def init_world ():
    world = None
    world = physics.World ()
    world.gravity = 0, 1

    shape = physics.RectShape ((0, 0, 20, 20))
    body = physics.Body (shape)
    body.position = 0, 0
    body.velocity = 2, 0
    body.restitution = 1
    body.mass = 20

    world.add_body (body)
    
    shape = physics.RectShape ((0, 0, 100, 10))
    base = physics.Body (shape)
    base.position = 0, 400
    base.static = True
    base.mass = 200000
    
    world.add_body (base)
    
    return world

def run ():
    video.init ()
    time.init ()
    
    world = init_world ()
    
    screen = video.set_mode (800, 600)
    screen.fill (black)
    screen.flip ()
    
    while True:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                sys.exit ()
            if ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                sys.exit ()
            screen.flip ()
        world.update (.2)
        time.delay (10)
        screen.fill (black)
        for body in world.bodies:
            render_body (screen, body)
        screen.flip ()
        
            
if __name__ == "__main__":
    run ()
