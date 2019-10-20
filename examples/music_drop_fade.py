import pygame as pg
import os, sys

def add_file(filename):
    '''
    This function will check if filename exists and is a music file
    If it is the file will be added to a list of music files
    The only type checking is by the extension of the file, not by its contents

    It looks in the file directory and its data subdirectory
    '''
    if filename.rpartition('.')[2] not in file_types: 
        print('only these files types are allowed: ',file_types)
        return False
    elif os.path.exists(filename):
        file_list.append(filename)
    elif os.path.exists(os.path.join(main_dir, filename)):
        file_list.append(os.path.join(main_dir, filename))
    elif os.path.exists(os.path.join(data_dir, filename)):
        file_list.append(os.path.join(data_dir, filename))
    else:
        print('file not found')
        return False
    print('{} added to file list'.format(filename))
    return True

def play_file(filename):
    '''
    This function will call add_file and play it if successful
    The music will fade in over the first 4 seconds
    set_endevent is used to post a MUSIC_DONE event when the song finishes
    The main loop will call play_next() when the MUSIC_DONE event is received
    '''
    if add_file(filename):
        try:
            pg.mixer.music.load(file_list[-1])
        except:
            print('{} is not a valid music file'.format(filename))
            return
        pg.mixer.music.play(fade_ms=4000)
        pg.mixer.music.set_endevent(MUSIC_DONE)

def play_next():
    '''
    This function is will play the next song in file_list
    It uses pop(0) to get the next song and then appends it to the end of the list
    The song will fade in over the first 4 seconds
    '''
    if len(file_list) > 1:
        nxt = file_list.pop(0)
        try:
            pg.mixer.music.load(nxt)
        except:
            print('{} is not a valid music file'.format(nxt))
            return

        file_list.append(nxt)
        print('starting next song: ', nxt)
    pg.mixer.music.play(fade_ms=4000)

def draw_line(text, y = 0):
    '''
    Draws a line of text onto the display surface
    The text will be centered horizontally at the given y postition
    The text's height is added to y and returned to the caller
    '''
    screen = pg.display.get_surface()
    surf = font.render(text,  1, (255, 255, 255))
    y += surf.get_height()
    x = (screen.get_width() - surf.get_width() ) / 2
    screen.blit(surf, (x,y))
    return y

MUSIC_DONE = pg.USEREVENT+1 # create a user event called MUSIC_DON in the first slot
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

file_list = []
file_types = ('mp3', 'ogg', 'mid', 'mod', 'it', 'xm')

def main():
    global font # this will be used by the draw_line function
    running = True
    paused = False

    pg.init()
    pg.display.set_mode((640, 480))
    font = pg.font.SysFont("Arial", 24)
    clock = pg.time.Clock()
    pg.scrap.init()
    pg.SCRAP_TEXT = pg.scrap.get_types()[0] # TODO update when the SDL2 scrap module is fixed
    clipped = pg.scrap.get(pg.SCRAP_TEXT)

    # add the command line arguments to the file_list
    for arg in sys.argv[1:]:
        add_file(arg)
    play_file('house_lo.ogg') # play default music included with pygame

    # draw instructions on screen
    y = draw_line("Drop music files or path names onto this window", 20)
    y = draw_line("Copy file names into the clipboard", y)
    y = draw_line("Or feed them from the command line", y)
    y = draw_line("If it's music it will play!", y)

    '''
    This is the main loop
    It will respond to drag and drop, clipboard changes, and key presses
    '''
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.DROPTEXT:
                print(ev)
                play_file(ev.text)
            elif ev.type == pg.DROPFILE:
                print(ev)
                play_file(ev.file)
            elif ev.type == MUSIC_DONE:
                play_next()
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False # exit loop
                elif ev.key in (pg.K_SPACE, pg.K_RETURN):
                    if paused:
                        pg.mixer.music.unpause()
                        paused = False
                    else:
                        pg.mixer.music.pause()
                        paused = True
                else:
                    play_next()

        new_text = pg.scrap.get(pg.SCRAP_TEXT).decode('UTF-8') # TODO update when SDL2 scrap is fixed
        if new_text != clipped: # has the clipboard changed?
            clipped = new_text 
            play_file(clipped) # try to play the file if it has

        pg.display.flip()
        clock.tick(3) # keep CPU use down by updating screen and checking events 3 times per second

    pg.quit()


if __name__ == "__main__":
    main()
