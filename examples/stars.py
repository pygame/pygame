#!/usr/bin/env python

import time, random, math

import pygame
from pygame.locals import *

#constants
WINSIZE = [640, 480]
WINCENTER = [320, 240]
NUMSTARS = 150


def init_star():
	"creates new star values"
	dir = random.randrange(100000)
	velmult = random.random()*.6+.4
	vel = [math.sin(dir) * velmult, math.cos(dir) * velmult]
	return vel, WINCENTER[:]


def initialize_stars():
	"creates a new starfield"
	stars = []
	for x in range(NUMSTARS):
		star = init_star()
		vel, pos = star
		steps = random.randint(0, WINCENTER[0])
		pos[0] += vel[0] * steps
		pos[1] += vel[1] * steps
		vel[0] *= steps * .09
		vel[1] *= steps * .09
		stars.append(star)
	move_stars(stars)
	return stars
	

def draw_stars(surface, stars, color):
	"used to draw (and clear) the stars"
	for vel, pos in stars:
		surface.set_at(pos, color)


def move_stars(stars):
	"animate the star values"
	for vel, pos in stars:
		pos[0] += vel[0]
		pos[1] += vel[1]
		if pos[0] < 0 or pos[1] < 0 or pos[0] >= WINSIZE[0] or pos[1] >= WINSIZE[1]:
			vel[:], pos[:] = init_star()
		else:
			vel[0] *= 1.05
			vel[1] *= 1.05
	

def main():
	"This is the starfield code"
	#create our starfield
	random.seed()
	stars = initialize_stars()

	#initialize and prepare screen
	pygame.init()
	screen = pygame.display.set_mode(WINSIZE)
	pygame.display.set_caption('pyGame Stars Example')
	white = screen.map_rgb(255, 240, 200)
	black = screen.map_rgb(20, 20, 40)
	screen.fill(black)

	#main game loop
	done = 0
	while not done:
		draw_stars(screen, stars, black)
		move_stars(stars)
		draw_stars(screen, stars, white)
		pygame.display.flip()

		for e in pygame.event.get():
			if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
				done = 1
				break
			elif e.type == MOUSEBUTTONDOWN and e.button == 1:
				WINCENTER[:] = list(e.pos)



# if python says run, then we should run
if __name__ == '__main__':
	main()


