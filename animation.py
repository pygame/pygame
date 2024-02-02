# This is a pygame module to create sprite animations

import pygame, time

# Load animation function
def load_animation(fileName, numSprites, extension, scale_x, scale_y, color):
    animation = []
    # For loop to iterate between in sprites
    for i in range(0, numSprites+1):
        file = fileName+str(i)+extension
        image = pygame.image.load(file)
        image.set_colorkey(color)
        image_scale = pygame.transform.scale(image, (scale_x, scale_y))
        animation.append(image_scale)
    
    return animation

# Show animation function
def show_animation(surface, images, x, y):
    frame = int(time.time()) % len(images) # Frame counter
    surface.blit(images[frame], (x, y)) # Display animation
