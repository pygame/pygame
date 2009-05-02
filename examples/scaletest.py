#!/usr/bin/env python

import sys, time
import pygame

def main(imagefile, convert_alpha=False, run_speed_test=False):
    """show an interactive image scaler

    arguemnts:
    imagefile - name of source image (required)
    convert_alpha - use convert_alpha() on the surf (default False)
    run_speed_test - (default False)

    """

    bSpeedTest = run_speed_test
    # initialize display
    pygame.display.init()
    # load background image
    background = pygame.image.load(imagefile)
    if convert_alpha:
        screen = pygame.display.set_mode((1024, 768), pygame.FULLSCREEN)
        background = background.convert_alpha()

    if bSpeedTest:
        SpeedTest(background)
        return
    screen = pygame.display.set_mode((1024, 768), pygame.FULLSCREEN)
    # start fullscreen mode
    # turn off the mouse pointer
    pygame.mouse.set_visible(0)
    # main loop
    bRunning = True
    bUp = False
    bDown = False
    bLeft = False
    bRight = False
    cursize = [background.get_width(), background.get_height()]
    while(bRunning):
        image = pygame.transform.smoothscale(background, cursize)
        imgpos = image.get_rect(centerx=512, centery=384)
        screen.fill((255,255,255))
        screen.blit(image, imgpos)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                bRunning = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    bUp = True
                if event.key == pygame.K_DOWN:  bDown = True
                if event.key == pygame.K_LEFT:  bLeft = True
                if event.key == pygame.K_RIGHT: bRight = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:    bUp = False
                if event.key == pygame.K_DOWN:  bDown = False
                if event.key == pygame.K_LEFT:  bLeft = False
                if event.key == pygame.K_RIGHT: bRight = False
        if bUp:
            cursize[1] -= 2
            if cursize[1] < 1: cursize[1] = 1
        if bDown:
            cursize[1] += 2
        if bLeft:
            cursize[0] -= 2
            if cursize[0] < 1: cursize[0] = 1
        if bRight:
            cursize[0] += 2


def SpeedTest(image):
    print ("Smoothscale Speed Test - Image Size %s\n" % str(image.get_size()))
    imgsize = [image.get_width(), image.get_height()]
    duration = 0.0
    for i in range(128):
        shrinkx = (imgsize[0] * i) / 128
        shrinky = (imgsize[1] * i) / 128
        start = time.time()
        tempimg = pygame.transform.smoothscale(image, (shrinkx, shrinky))
        duration += (time.time() - start)
        del tempimg
    print ("Average smooth shrink time: %i milliseconds." % int((duration / 128) * 1000))
    duration = 0
    for i in range(128):
        expandx = (imgsize[0] * (i + 129)) / 128
        expandy = (imgsize[1] * (i + 129)) / 128
        start = time.time()
        tempimg = pygame.transform.smoothscale(image, (expandx, expandy))
        duration += (time.time() - start)
        del tempimg
    print ("Average smooth expand time: %i milliseconds." % int((duration / 128) * 1000))
    duration = 0.0
    for i in range(128):
        shrinkx = (imgsize[0] * i) / 128
        shrinky = (imgsize[1] * i) / 128
        start = time.time()
        tempimg = pygame.transform.scale(image, (shrinkx, shrinky))
        duration += (time.time() - start)
        del tempimg
    print ("Average jaggy shrink time: %i milliseconds." % int((duration / 128) * 1000))
    duration = 0
    for i in range(128):
        expandx = (imgsize[0] * (i + 129)) / 128
        expandy = (imgsize[1] * (i + 129)) / 128
        start = time.time()
        tempimg = pygame.transform.scale(image, (expandx, expandy))
        duration += (time.time() - start)
        del tempimg
    print ("Average jaggy expand time: %i milliseconds." % int((duration / 128) * 1000))



if __name__ == '__main__':
    # check input parameters
    if len(sys.argv) < 2:
        print ("Usage: %s ImageFile [-t] [-convert_alpha]" % sys.argv[0])
        print ("       [-t] = Run Speed Test\n")
        print ("       [-convert_alpha] = Use convert_alpha() on the surf.\n")
    else:
        main(sys.argv[1],
             convert_alpha = '-convert_alpha' in sys.argv,
             run_speed_test = '-t' in sys.argv)
