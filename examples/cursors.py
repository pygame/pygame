#!/usr/bin/env python
""" pygame.examples.cursors

Click a button and the cursor will change.

This example will show you:
*The different types of cursors that exist
*How to create a cursor
*How to set a cursor

"""

import pygame as pg
import os


#Create a system cursor

system_cursor = pg.SYSTEM_CURSOR_CROSSHAIR


#Create color cursor

surf = pg.Surface((40, 40)) 
surf.fill((120, 50, 50))     
color_cursor = pg.cursors.Cursor((20, 20), surf)


#Create a color cursor with an image surface

main_dir = os.path.split(os.path.abspath(__file__))[0]
image_name = os.path.join(main_dir, "data", "cursor.png")
image = pg.image.load(image_name)
image_cursor  = pg.cursors.Cursor((image.get_width()//2, image.get_height()//2), image)


#Create bitmap cursor from simple strings

# sized 24x24
thickarrow_strings = (
    "XX                      ",
    "XXX                     ",
    "XXXX                    ",
    "XX.XX                   ",
    "XX..XX                  ",
    "XX...XX                 ",
    "XX....XX                ",
    "XX.....XX               ",
    "XX......XX              ",
    "XX.......XX             ",
    "XX........XX            ",
    "XX........XXX           ",
    "XX......XXXXX           ",
    "XX.XXX..XX              ",
    "XXXX XX..XX             ",
    "XX   XX..XX             ",
    "     XX..XX             ",
    "      XX..XX            ",
    "      XX..XX            ",
    "       XXXX             ",
    "       XX               ",
    "                        ",
    "                        ",
    "                        ",
)

bitmap_cursor1 = pg.cursors.Cursor(
    (24, 24), (0, 0), *pg.cursors.compile(thickarrow_strings, black='X', white='.', xor='o') 
)


#Create bitmap cursor from premade simple strings

bitmap_cursor2 = pg.cursors.diamond


#Calculate if mouse position is inside circle

def check_circle(mouse_pos_x, mouse_pos_y, center_x, center_y, radius):
    return (mouse_pos_x - center_x) ** 2 + (mouse_pos_y - center_y) ** 2 < radius ** 2

def main():
    pg.init()
    pg.font.init()
    font = pg.font.Font(None, 24)

    bg = pg.display.set_mode((600, 400))
    bg.fill((225,225,225))


    text2 = font.render("Hover over the circles to change their color!", True, (0, 0, 0))
    text_rect2 = text2.get_rect(center=(bg.get_width()/2, 100))
    bg.blit(text2, text_rect2)

    #Initialize circles
    radius = 60
    circle_red = pg.draw.circle(bg, (255, 255, 255), (100, 200), radius)
    circle_green = pg.draw.circle(bg, (255, 255, 255), (300, 200), radius)
    circle_blue = pg.draw.circle(bg, (255, 255, 255), (500, 200), radius)

    #Initialize button
    button_text = font.render("Click here to change the cursor", True, (0, 0, 0))
    button = pg.draw.rect(bg, (180,180,180),(175, 300, button_text.get_width()+5, button_text.get_height() + 50))
    button_text_rect = button_text.get_rect(center=(button.center))
    bg.blit(button_text, button_text_rect)

    pg.display.set_caption("Cursors Example")

    pg.display.update()


    cursors = [system_cursor, color_cursor, image_cursor, bitmap_cursor1, bitmap_cursor2]
    index = 0
    pg.mouse.set_cursor(cursors[index])

    
    while True:

        mouse_pos = pg.mouse.get_pos
        mouse_x = mouse_pos()[0]
        mouse_y = mouse_pos()[1]

        if check_circle(mouse_x, mouse_y, circle_red.centerx, circle_red.centery, radius):
            circle_red = pg.draw.circle(bg, (255, 0, 0), (100, 200), radius)
     
        elif check_circle(mouse_x, mouse_y, circle_green.centerx, circle_green.centery, radius):
            circle_green = pg.draw.circle(bg, (0, 255, 0), (300, 200), radius)
       
        elif check_circle(mouse_x, mouse_y, circle_blue.centerx, circle_blue.centery, radius):
            circle_blue = pg.draw.circle(bg, (0, 0, 255), (500, 200), radius)
      
        else:
            circle_red = pg.draw.circle(bg, (255, 255, 255), (100, 200), radius)
            circle_green = pg.draw.circle(bg, (255, 255, 255), (300, 200), radius)
            circle_blue = pg.draw.circle(bg, (255, 255, 255), (500, 200), radius)
      

        bg.fill((200,200,200), (0, 15 ,bg.get_width(), 50))

        text1 = font.render(("This is a "+ pg.mouse.get_cursor().type +" cursor"), True, (0, 0, 0))
        text_rect1 = text1.get_rect(center=(bg.get_width()/2, 40))

        bg.blit(text1, text_rect1)

        
        for event in pg.event.get():

            button = pg.draw.rect(bg, (180,180,180),(175, 300, button_text.get_width()+5, button_text.get_height() + 50))
            bg.blit(button_text, button_text_rect)

            if mouse_x > button.x and mouse_x < button.x + button.width and mouse_y > button.y and mouse_y < button.y + button.height:
                button = pg.draw.rect(bg, (120,120,120),(175, 300, button_text.get_width()+5, button_text.get_height() + 50))
                bg.blit(button_text, button_text_rect)

                if pg.mouse.get_pressed()[0]:
                    button = pg.draw.rect(bg, (50,50,50),(175, 300, button_text.get_width()+5, button_text.get_height() + 50))
                    bg.blit(button_text, button_text_rect)
                    index += 1
                    index %= len(cursors)
                    pg.mouse.set_cursor(cursors[index])   
                
            pg.display.update()
        
            if event.type == pg.QUIT:
                pg.quit()
                raise SystemExit




if __name__ == "__main__":
    main()
