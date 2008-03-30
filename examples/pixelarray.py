import os, pygame

pygame.init ()

screen = pygame.display.set_mode ((255, 255))
surface = pygame.Surface ((255, 255))

pygame.display.flip ()

def show (image):
    screen.fill ((0, 0, 0))
    screen.blit (image, (0, 0))
    pygame.display.flip ()
    while 1:
        event = pygame.event.wait ()
        if event.type == pygame.QUIT:
            raise SystemExit
        if event.type == pygame.MOUSEBUTTONDOWN:
            break

# Create the PixelArray.
ar = pygame.PixelArray (surface)
r, g, b = 0, 0, 0
# Do some easy gradient effect.
for y in xrange (255):
    r, g, b = y, y, y
    ar[:,y] = (r, g, b)
del ar
show (surface)

# We have made some gradient effect, now flip it.
ar = pygame.PixelArray (surface)
ar[:] = ar[:,::-1]
del ar
show (surface)

# Every second column will be made blue
ar = pygame.PixelArray (surface)
ar[::2] = (0, 0, 255)
del ar
show (surface)

# Every second row will be made green
ar = pygame.PixelArray (surface)
ar[:,::2] = (0, 255, 0)
del ar
show (surface)

# Manipulate the image. Flip it around the y axis.
surface = pygame.image.load (os.path.join ('data', 'arraydemo.bmp'))
ar = pygame.PixelArray (surface)
ar[:] = ar[:,::-1]
del ar
show (surface)

# Flip the image around the x axis.
ar = pygame.PixelArray (surface)
ar[:] = ar[::-1,:]
del ar
show (surface)

# Every second column will be made white.
ar = pygame.PixelArray (surface)
ar[::2] = (255, 255, 255)
del ar
show (surface)

# Flip the image around both axes, restoring it's original layout.
ar = pygame.PixelArray (surface)
ar[:] = ar[::-1,::-1]
del ar
show (surface)

surface = pygame.image.load (os.path.join ('data', 'arraydemo.bmp'))
ar = pygame.PixelArray (surface)
sf2 = ar[::2,::2].make_surface ()
del ar
show (sf2)
