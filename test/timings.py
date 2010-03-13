import timeit

fillfuncmap = {
    'BLEND_RGBA_ADD' : """
    fill (c2, blendargs=const.BLEND_RGBA_ADD)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_AND' : """
    fill (c2, blendargs=const.BLEND_RGBA_AND)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_AVG' : """
    fill (c2, blendargs=const.BLEND_RGBA_AVG)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_DIFF' : """
    fill (c2, blendargs=const.BLEND_RGBA_DIFF)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_MAX' : """
    fill (c2, blendargs=const.BLEND_RGBA_MAX)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_MIN' : """
    fill (c2, blendargs=const.BLEND_RGBA_MIN)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_MULT' : """
    fill (c2, blendargs=const.BLEND_RGBA_MULT)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_OR' : """
    fill (c2, blendargs=const.BLEND_RGBA_OR)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_SCREEN' : """
    fill (c2, blendargs=const.BLEND_RGBA_SCREEN)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_SUB' : """
    fill (c2, blendargs=const.BLEND_RGBA_SUB)
    c1, c2 = c2, c1
    """,
    'BLEND_RGBA_XOR' : """
    fill (c2, blendargs=const.BLEND_RGBA_XOR)
    c1, c2 = c2, c1
    """,
    }   

fillsetupstr = """
from pygame2 import Color
import pygame2.sdl.video as video
import pygame2.sdl.constants as const

c1 = Color (250, 126, 2)
c2 = Color (2, 126, 250)

video.init ()
sf = video.Surface (500, 500, 32, const.SRCALPHA)
sf.fill (c2)
fill = sf.fill
"""

blendfuncmap = {
    'BLEND_RGBA_ADD' : """
    blit (sf, blendargs=const.BLEND_RGBA_ADD)
    fill (color)
    """,
    'BLEND_RGBA_AND' : """
    blit (sf, blendargs=const.BLEND_RGBA_AND)
    fill (color)
    """,
    'BLEND_RGBA_AVG' : """
    blit (sf, blendargs=const.BLEND_RGBA_AVG)
    fill (color)
    """,
    'BLEND_RGBA_DIFF' : """
    blit (sf, blendargs=const.BLEND_RGBA_DIFF)
    fill (color)
    """,
    'BLEND_RGBA_MAX' : """
    blit (sf, blendargs=const.BLEND_RGBA_MAX)
    fill (color)
    """,
    'BLEND_RGBA_MIN' : """
    blit (sf, blendargs=const.BLEND_RGBA_MIN)
    fill (color)
    """,
    'BLEND_RGBA_MULT' : """
    blit (sf, blendargs=const.BLEND_RGBA_MULT)
    fill (color)
    """,
    'BLEND_RGBA_OR' : """
    blit (sf, blendargs=const.BLEND_RGBA_OR)
    fill (color)
    """,
    'BLEND_RGBA_SCREEN' : """
    blit (sf, blendargs=const.BLEND_RGBA_SCREEN)
    fill (color)
    """,
    'BLEND_RGBA_SUB' : """
    blit (sf, blendargs=const.BLEND_RGBA_SUB)
    fill (color)
    """,
    'BLEND_RGBA_XOR' : """
    blit (sf, blendargs=const.BLEND_RGBA_XOR)
    fill (color)
    """,
    }   

blendsetupstr = """
from pygame2 import Color
import pygame2.sdl.video as video
import pygame2.sdl.constants as const

color = Color (170, 126, 23)

video.init ()
surface = video.Surface (500, 500, 32, const.SRCALPHA)
sf = video.Surface (400, 400, 32, const.SRCALPHA)
surface.fill (color)
fill = surface.fill
blit = surface.blit
"""

print ("SDL Surface fill functions")
print ("##########################")
for n, f in fillfuncmap.items():
    print ("%s:\t %f" % (n , timeit.timeit (f, fillsetupstr, number=1000)))
print ("##########################")
print ("SDL Surface blit functions")
print ("##########################")
for n, f in blendfuncmap.items():
    print ("%s:\t %f" % (n , timeit.timeit (f, blendsetupstr, number=1000)))
print ("##########################")
