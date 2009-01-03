# Quick and dirty demo

import sys
import os

BLACK = 0, 0, 0
DISPLAY_SIZE = 240, 320

if os.name == "e32":
    f=open('/data/pygame/stdout.txt','w')
    f.write('SDL!\n')
    sys.stdout = f
    sys.stderr = f
    
    THISDIR = "\\data\\pygame"
    
else:
    THISDIR = os.path.dirname( __file__ )

import pygame

surfaces = []

def fade( screen, clock, image, imgrect, steps = 15, fadein = True ):            
    # Center the image        
    jump = 256. / steps
     
    for x in xrange( 1, ( steps + 1) ):
        x = x*jump
        if not fadein:
            x = 256 - x
        
        clock.tick(30)
        image.set_alpha( x )
        
        screen.fill(BLACK)
        for img, rect in surfaces:
            screen.blit(img, rect )
        pygame.display.flip()
        
    return imgrect
    
def imagefade(screen, clock, image, posy, delay = 100, fadeout = True ):
    
    imgrect = image.get_rect()            
    surfaces.append( ( image, imgrect ) )
    
    posx = pygame.display.Info().current_w / 2 - imgrect.w / 2
    posy = posy - imgrect.h / 2
    #posy = pygame.display.Info().current_h / 2 - imgrect.h / 2
    
    imgrect.x = posx
    imgrect.y = posy
    
    imgrect = fade( screen, clock, image, imgrect  )
    
    pygame.time.delay(delay)
    
    if fadeout:
        imgrect = fade( screen, clock, image, imgrect, fadein = False )
    
    return image, imgrect

def show_msg( screen, clock, text ):
    
    font = pygame.font.Font(None, 20)
    size = font.size(text)
    print "size", size
    fg = 0, 200, 0
    bg = 5, 5, 5
    
    ren = font.render(text, 0, fg, bg)  
    
    rect = ren.get_rect()    
        
    steps = 30
    jump = 256. / steps
    
    angle_jump = 90. / steps * 2
    
    for x in xrange( 1, ( steps + 1) ):
        if x < steps / 2:
            img  = pygame.transform.rotate( ren, 90 - angle_jump * x )        
        else:
            img = ren
        
        x = x*jump        
        x = 256 - x
        clock.tick(30)
        
        screen.fill(BLACK)
        
        
        img.set_alpha( x )
        
        rect = img.get_rect()
        posx = pygame.display.Info().current_w / 2 - rect.w / 2
        posy = pygame.display.Info().current_h / 2 - rect.h / 2
        
        rect.x = posx
        rect.y = posy
    
        screen.blit(img, rect )
        
        pygame.display.flip()
     
        
def intro():
    
    pygame.display.init()    
    pygame.font.init()
    
    print pygame.display.list_modes()
    size = width, height = pygame.display.list_modes()[0]
    screen = None
    if os.name == "e32":
        try:
            screen = pygame.display.set_mode(size)
        except pygame.error:
            screen = pygame.display.set_mode( (32,32) )#DISPLAY_SIZE)
    else:
        screen = pygame.display.set_mode( DISPLAY_SIZE )
        
    clock = pygame.time.Clock()    
    
    global show_msg
    text = 'Pygame Demo loading'
    show_msg( screen, clock, text )
    
    pygame.mixer.init( 22050, -32, 0, 4094)    
    
    # Init the rest
    pygame.init()       
    
    show_msg = False
    
    img_sdl    = pygame.image.load( os.path.join( THISDIR, "sdl_powered.bmp") )
    img_python = pygame.image.load( os.path.join( THISDIR, "python_powered.bmp") )
    img_pygame = pygame.image.load( os.path.join( THISDIR, "pygame_powered.bmp") )
    
    s1 = pygame.mixer.Sound( THISDIR + "\\ambient.ogg" )
    #s2 = pygame.mixer.Sound( THISDIR + "\\piano.ogg" )
    
    s1.set_volume( 0.1 )
    #s2.set_volume( 0.1 )
        
    speed = [2, 2]
        
    #s2.play()
    posy = pygame.display.Info().current_h / 9
    s1.play()
    imagefade( screen, clock, img_sdl, posy * 2, fadeout=False )
    imagefade( screen, clock, img_python, posy * 6, fadeout=False )
    imagefade( screen, clock, img_pygame, posy * 4, fadeout=False )
     
    global surfaces
    pygame_img = surfaces[-1]
    surfaces = surfaces[:-1]
    
    steps = 15
    jump = 256. / steps     
    for x in xrange( 1, ( steps + 1) ):
        x = x*jump        
        x = 256 - x
        
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.scancode == 165:
                    running = 0
                elif event.key == pygame.constants.K_ESCAPE:
                    running = 0
            elif event.type == pygame.QUIT:
                running = 0
            print event
                        
        screen.fill(BLACK)
        
        img, rect = pygame_img        
        screen.blit(img, rect ) 
        
        for img, rect in surfaces:
            img.set_alpha( x )
            screen.blit(img, rect )
        pygame.display.flip()
    
    s1.fadeout(1500)
    #s2.fadeout(500)
    pygame.time.delay( 1500 )
    

    steps = 15
    jump = 256. / steps
    for i in xrange( 1, ( steps + 1) ):
        x = i*jump        
        x = 256 - x
        
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.scancode == 165:
                    running = 0
                elif event.key == pygame.constants.K_ESCAPE:
                    running = 0
            elif event.type == pygame.QUIT:
                running = 0
                        
        screen.fill(BLACK)
        img, rect = pygame_img
        
        scale = 1.0 - i / float(steps)       
        im  = pygame.transform.scale( img, (rect.w * scale, rect.h * scale) )       
        im.set_alpha( x )
        screen.blit(im, rect )
        pygame.display.flip()
    
    pygame.time.delay( 500 )
    
intro()

if os.name != "nt":
    f.close()

