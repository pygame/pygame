"""
This example demonstrates how to find fonts for each language.
"""

from typing import Iterator

from pygame.font import get_fonts
from pygame import freetype
import pygame


def enum_font_from_lang(lang) -> Iterator[str]:
    '''
    Enumerates fonts that are suitable for a specified language.
    '''
    text = DISCRIMINANTS[lang]
    for font_name in get_fonts():
        if can_render_text(font_name, text):
            yield font_name


def can_render_text(font_name, text, *, font_size=14) -> bool:
    '''
    Whether a specified font is capable of rendering a specified text.
    '''
    try:
        font = freetype.SysFont(font_name, font_size)
    except AttributeError as e:
        # AttributeError occurs on certain fonts, which may be a PyGame bug.
        if e.args[0] == r"this style is unsupported for a bitmap font":
            return False
        else:
            raise
    rendered = []
    for c in text:
        pixels = font.render_raw(c)
        if pixels in rendered:
            return False
        rendered.append(pixels)
    return True


DISCRIMINANTS = {
    'ja': '経伝説あAB',
    'ko': '안녕조AB',
    'zh-Hans': '哪经传说AB',
    'zh-Hant': '哪經傳說AB',
    'zh': '哪經傳說经传说AB',
}
'''
You may have noticed the 'AB' in the values of this dictionary.
If you don't include ASCII characters, you might choose a font that cannot render them, such as a fallback font,
which is probably not what you want.
'''


def main():
    pygame.init()

    screen = pygame.display.set_mode((800, 800))
    screen.fill("white")
    x, y = 40, 10
    spacing = 10
    for font_name in enum_font_from_lang('ja'):
        font = freetype.SysFont(font_name, 30)
        rect = font.render_to(screen, (x, y), "経伝説 한글 经传说 ABC 經傳說 あいう")
        y = y + rect.height + spacing
    pygame.display.flip()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        clock.tick(4)

    pygame.quit()


if __name__ == "__main__":
    main()
