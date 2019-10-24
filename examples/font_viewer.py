#! /usr/bin/env python3

import sys
import os
import pygame as pg
from pygame.constants import *

class fontviewer:
	KEY_SPEED = 10

	def __init__(self, directory = None):
		pg.init()
		self.font_dir = directory
		info = pg.display.Info()
		w = info.current_w
		h = info.current_h
		pg.display.set_mode((int(w * 0.8), int(h * 0.8)))
		self.clock = pg.time.Clock()
		self.y_offset = 0
		self.grabbed = False
		self.Render('&N abcDEF789', size=48, color = (0,0,0), back_color = (255,255,255))
		self.Display()
		#viewer.Save()
		#pg.mouse.set_visible(not self.grabbed)
		#pg.event.set_grab(self.grabbed)

	def Render(self, text = 'A display of font &N', **dparams):
		size = dparams.get('size', 32)
		color = dparams.get('color', (255,255,255))
		self.back_color = dparams.get('back_color', (0,0,0))

		fonts = pg.font.get_fonts()
		surfaces = []
		total_height = 0
		max_width = 0
		
		font = pg.font.SysFont(pg.font.get_default_font(), size)
		lines = (
			'Click in this window to enter scroll mode',
			'The mouse will be grabbed and hidden until you click again',
			'Some fonts might be rendered incorrectly',
			'Here are you system fonts')
		for line in lines:
			surf = font.render(line, 1, color, self.back_color)

			total_height += surf.get_height()
			max_width = max(max_width, surf.get_width())
			surfaces.append(surf)

		print('found {} fonts on your system'.format(len(fonts)))
		for name in sorted(fonts):
			print(name)
			font = pg.font.SysFont(name, size)
			line = text.replace('&N', name)
			surf = font.render(line, 1, color, self.back_color)

			total_height += surf.get_height()
			max_width = max(max_width, surf.get_width())
			surfaces.append(surf)
		rendered = pg.surface.Surface((max_width, total_height))
		rendered.fill(self.back_color)
		print('scrolling surface created')
		print('total height={}, max width={}'.format(total_height, max_width))

		y = 0
		center = int(max_width / 2)
		for surf in surfaces:
			w = surf.get_width()
			c = center - int(w/2)
			rendered.blit(surf, (c, y))
			y+= surf.get_height()
		self.surface = rendered.convert()
		self.max_y = rendered.get_height() - pg.display.get_surface().get_height()

	def Save(self, name = 'fontviewer.png'):
		pg.image.save(self.surface, name)

	def Display(self, time = 10):
		screen = pg.display.get_surface()
		rect = pg.rect.Rect(0, 0, self.surface.get_width(),
				min( self.surface.get_height(), screen.get_height()) )
		print(rect)
		x = int( (screen.get_width() - self.surface.get_width()) / 2)
		clr = pg.rect.Rect(x, 0, self.surface.get_width(), screen.get_height())

		while True:
			if not self.HandleEvents():
				break
			screen.fill( self.back_color )
			rect.y = self.y_offset
			screen.blit( self.surface, (x, 0), rect)
			pg.display.flip()
			self.clock.tick(20)

	def HandleEvents(self):
		events = pg.event.get()
		for e in events:
			if e.type == QUIT:
				return False
			elif e.type == KEYDOWN:
				if e.key == K_ESCAPE:
					return False
			elif e.type ==  MOUSEBUTTONDOWN:
				self.grabbed = not self.grabbed
				pg.event.set_grab(self.grabbed)
				pg.mouse.set_visible(not self.grabbed)
		keys = pg.key.get_pressed()
		if keys[K_UP]:
			self.key_held += 1
			self.y_offset -= self.KEY_SPEED * (self.key_held / 10)
		elif keys[K_DOWN]:
			self.key_held += 1
			self.y_offset += self.KEY_SPEED * (self.key_held / 10)
		else:
			self.key_held = 20

		y = pg.mouse.get_rel()[1]
		if y and self.grabbed:
			self.y_offset += (y/2) ** 2 * (y/abs(y))
			self.y_offset = min((max(self.y_offset, 0), self.max_y))
		return True


viewer = fontviewer()

