#/usr/bin/env python

"""Running this script will give a slideshow type
performance from all the images inside a .ZIP
file. pass a zipfile as an argument, or it will
try to find one in the current directory.

Also note, if your SDL_image doesn't support
.JPG, you won't see all the images available"""

#note, if not using SDL_image-1.1.0 or greater,
#you will likely crash when loading a non-image
#(don't blame me, it's a bug in older SDL_image)


import zipfile, sys, glob, os
import pygame, pygame.image
from cStringIO import StringIO
from pygame.locals import *


def showimage(imgsurface, name):
    "put a loaded image onto the screen"
    size = imgsurface.get_size()
    pygame.display.set_caption(name+'  '+`size`)
    screen = pygame.display.set_mode(size)
    screen.blit(imgsurface, (0,0))
    pygame.display.flip()



def zipslideshow(zipfilename):    
    "loop through all the images in the zipfile"
    pygame.init()
    
    zip = zipfile.ZipFile(zipfilename)
    for file in zip.namelist():
        #get data from zipfile
        data = zip.read(file)
        #create stringio for data (a file-like object)
        data_io = StringIO(data)
        #load from a stringio object
        try: surf = pygame.image.load(data_io, file)
        except: continue
        #get image on the screen
        showimage(surf, file)

        #loop through events until done with this pic
        finished = 0
        while 1:
            e = pygame.event.wait()
            if e.type == QUIT:
                finished = 1
                break
            if e.type in (KEYDOWN, MOUSEBUTTONDOWN):
                break
        if finished:
            break



def main():
    "run the program, handle arguments"
    zipfiles = sys.argv[1:2]
    if not zipfiles:
        zipfiles = glob.glob('*.zip')
    if not zipfiles:
        raise SystemExit('No ZIP files given, or in current directory')
    zipslideshow(zipfiles[0])


#run the script if not being imported
if __name__ == '__main__': main()

