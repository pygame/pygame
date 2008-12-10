import sys

import trackmod
trackmod.begin('playmus_rep.txt')
import pygame

MusicEnd = pygame.USEREVENT

def main():
    pygame.mixer.init()
    pygame.mixer.music.set_endevent(MusicEnd)
    pygame.mixer.music.load(sys.argv[1])
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.quit()

if __name__ == '__main__':
    main()

