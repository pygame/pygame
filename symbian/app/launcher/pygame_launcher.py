"""
pygame launcher for S60 - Application for launching others
"""

from glob import glob

import math

import sys
import os

import pygame
from pygame.locals import *
from pygame import constants

BLACK = 0, 0, 0, 255
WHITE = 255,255,255,255
TITLE_BG     = 0, 255, 0, 75
TITLE_STROKE = 0, 255, 0, 200
DISPLAY_SIZE = 240, 320
MENU_BG = 0, 255, 0, 75
ITEM_UNSELECTED_TEXT = 0, 128, 0, 128
ITEM_SELECTED_TEXT = 0, 255, 0, 128
ITEM_UNSELECTED = 0, 128, 0, 128
ITEM_SELECTED   = TITLE_BG

THISDIR = os.path.dirname( __file__ )

def load_image(name, colorkey=None):
    """ Image loading utility from chimp.py """
    fullname = os.path.join(THISDIR, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error, message:
        print 'Cannot load image:', name
        raise SystemExit, message
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

class SystemData:
    """ Common resources """
    def __init__(self):
        self.screen   = None
        self.ticdiff  = 0
        self.tics     = 0
        
    def get_font_title(self):
        return pygame.font.Font(None, 36)
    def get_font_normal(self):
        return pygame.font.Font(None, 25)
        #self.font_normal   = self.font_title#
    def get_font_normal_b(self):
        f = self.get_font_normal()
        f.set_bold(True)
        return f
    def get_font_small(self):
        return pygame.font.Font(None, 18)
        
class Background(pygame.sprite.Sprite):
    """Main background"""
    def __init__(self, sysdata):
        pygame.sprite.Sprite.__init__(self) 
        
        self.sysdata = sysdata
        
        self.screen = sysdata.screen
        screen_size = self.screen.get_size()
        #middle = #screen_size[0] / 2., screen_size[1] / 2.
        
        self.background = pygame.Surface(screen_size, SRCALPHA)
        self.background.fill(BLACK)
        
        self.rect       = self.background.get_rect()
        middle = self.rect.center
        
        self.img_logo, r = load_image( "logo.jpg")
        self.img_pos = [ middle[0] - r.w / 2, middle[1] - r.h / 2 ]
        
        self.alpha = pygame.Surface(r.size, SRCALPHA)
        
        self.alphaval = 200.
        self.alphadir = -20. # per second
      
    def update_alphaval(self):
        """ Update the visibility of the logo """
        min = 200.
        max = 245.
        
        s = self.sysdata.ticdiff / 1000.
        
        self.alphaval += s * self.alphadir
        # + (abs(self.alphadir) / self.alphadir) * ( (max - self.alphaval) / ( max -min )) 
        
        if self.alphaval > max:
            self.alphaval = max
            self.alphadir = -self.alphadir
            
        if self.alphaval < min:
            self.alphaval = min
            self.alphadir = -self.alphadir
        
    def update(self):
        
        self.update_alphaval()
        
        self.background.fill(BLACK)
        
        self.alpha.fill( (0,0,0,self.alphaval) )
        
        self.background.blit( self.img_logo, self.img_pos )
        self.background.blit(self.alpha, self.img_pos )
        

class TextField(pygame.sprite.Sprite):
    """ Shows text """
    MODE_NONE   = 0
    MODE_CENTER = 1
    
    def __init__(self, parent, sysdata, exit_callback, title = "", text = "", mode = MODE_NONE ):
        pygame.sprite.Sprite.__init__(self)
        
        self.sysdata = sysdata
        self.parent = parent
        self._title = title
        self.title_changed = True
        
        self._text = text
        self.text_changed = True
        self._selected_index = 0
        
        self.mode = mode
        
        self.exit_callback = exit_callback
        
    def update_text(self):
        """ Redraw text contents """
        if not self.text_changed: return
        
        text = self._text
        
        # Below title
        startposy = self.titlesurface.get_size()[1] + 20
        startposx = 10
        
        # Position on parent
        self.itemspos = ( 0, startposy )
        
        psize = self.parent.get_size()
        size = ( psize[0], psize[1] - startposy )
        surf = pygame.Surface(size, SRCALPHA)
        
        # Create text contents
        rect = pygame.Rect( 5, 0, size[0]-10, size[1]-5)
        pygame.draw.rect(surf, MENU_BG, rect)
        pygame.draw.rect(surf, TITLE_STROKE, rect, 2)
        
        #text = textwrap.dedent(text)
        font = self.sysdata.get_font_small()
        
        lines = text.split("\n")
        height = font.get_height()
        posy  = height
        for line in lines:
            line = line.strip()
            text = font.render(line, 1, (0,255,0) )
            if self.mode == TextField.MODE_CENTER:
                x,y = text.get_size()
                x = size[0] - x
                x /= 2
            
            surf.blit( text, (x,posy) )
            posy += height
        
        self.textsurface = surf
        self.text_changed = False
        
    def update_title(self):
        """ Redraw title text """
        if not self.title_changed: return
            
        text = self.sysdata.get_font_title().render(self._title, 1, (0, 255, 0))
        textpos = text.get_rect()
        textpos.centerx = self.parent.get_rect().centerx
        textpos.centery = textpos.size[1] / 2 + 7
        
        self.size = size = self.parent.get_size()
        size = ( size[0], 40 )

        surf = titlebg = pygame.Surface(size, SRCALPHA)
        self.titlebgrect = titlebgrect = pygame.Rect( 5, 5, size[0]-10, size[1]-10)
        pygame.draw.rect(surf, TITLE_BG, titlebgrect)
        pygame.draw.rect(surf, TITLE_STROKE, titlebgrect, 2)
        surf.blit(text, textpos )
        
        self.title_changed = False
        self.titlesurface = surf
        # Position on parent
        self.titlepos = (0,0)
        
    def update(self):
        self.update_title()
        self.update_text()
        
        self.parent.blit(self.titlesurface, self.titlepos )
        self.parent.blit(self.textsurface, self.itemspos )
    
    def exit(self):
        self.exit_callback()
        
    def handleEvent(self, event ):
        """ Exit on any key """
        if event.type == pygame.KEYDOWN:
            self.exit()
            return True
        
        return False
    
class Menu(pygame.sprite.Sprite):
    
    def __init__(self, parent, sysdata, title, items, cancel_callback):
        pygame.sprite.Sprite.__init__(self)
        
        self.sysdata = sysdata
        self.parent = parent
        self._title = title
        self.title_changed = True
        
        self._items = items
        self.items_changed = True
        self._selected_index = 0
        
        #: Index of the topmost visible item
        self.visibletop = 0
        #: How many items the list can display
        self.shownitems = 0
        
        #: Callback called at exit
        self.cancel_callback = cancel_callback
        print cancel_callback
        
    def _set_selection(self,index):
        self._selected_index = index
        self.items_changed = True
    
    def _get_selection(self): return self._selected_index
    selection = property(fget=_get_selection, fset=_set_selection )
    
    def select_next_item(self):
        """ Select next item from list. Loops the items """
        last = len(self._items) - 1
        if last < 1: return
        
        next = self._selected_index + 1
        if next > last:
            next = 0
        self.selection = next
        
        self.update_visible()
        
    def select_prev_item(self):
        """ Select previous item from list. Loops the items """
        last = len(self._items) - 1
        if last < 1: return
        
        first = 0
        next = self._selected_index - 1
        if next < first:
            next = last
        self.selection = next
        
        self.update_visible()
    
    def update_visible(self):
        """ Updates position of the topmost visible item in the list """
        diff = abs(self.visibletop - self._selected_index)
        if diff >= self.shownitems:
            self.visibletop = max(0,min( self._selected_index - self.shownitems+1, self._selected_index ))
        
    def select_item(self):
        """ Handle item selection by invoking its callback function """
        title, callback,args = self._items[self._selected_index]
        callback(*args)
    
    def _set_title(self, title):
        self._title = title
        self.title_changed = True
        
    def _get_title(self): return self._title
    title = property(fget=_get_title, fset=_set_title)
    
    def _set_items(self, title): 
        self._items = items
        self.items_changed = True
        
    def _get_items(self): return self._items
    items = property(fget=_get_items, fset=_set_items)
    
    def cancel(self):
        cb,args = self.cancel_callback
        cb(*args)
    
    def handleEvent(self, event ):
        if event.type == pygame.KEYDOWN:
            if event.key == constants.K_DOWN:
                self.select_next_item()
                return True
            
            elif event.key == constants.K_UP:
                self.select_prev_item()
                return True
            
            if event.key == constants.K_RETURN:
                self.select_item()
                return True
            
            if event.key == constants.K_ESCAPE:
                self.cancel()
                return True
            
        return False
    
    def clear(self):
        " Remove the used surfaces from memory "
        self.itemsurface = None
        self.titlesurface = None
        self.items_changed = True
        self.title_changed = True
    
    def max_items_shown(self):
        """ Calculate the amount of items that fit inside the list """

        height = self.sysdata.get_font_normal().get_height() * 1.5
        h = self.itemsurface.get_height()
        return int(round((h / height))) - 1
     
    def update_items(self):
        """ Update list item surface """
        if not self.items_changed: return
        
        items = self._items
        
        # Below title
        startposy = self.titlesurface.get_size()[1] + 20
        startposx = 10
        self.itemspos = ( 0, startposy )
        
        psize = self.parent.get_size()
        size = ( psize[0], psize[1] - startposy)
        surf = pygame.Surface(size, SRCALPHA)
        self.itemsurface = surf
        self.shownitems  = self.max_items_shown()
        
        # Create and cache the list background
        rect = pygame.Rect( 5, 0, size[0]-10, size[1]-5)
        
        pygame.draw.rect(surf, MENU_BG, rect)
        pygame.draw.rect(surf, TITLE_STROKE, rect, 2)
        
        startposy = 0
        
        self.visibletop = min(self.visibletop, self._selected_index)
        maximumpos = min(len(items), self.visibletop + self.shownitems )        
        height = self.sysdata.get_font_normal().get_height()
        spaceheight = height + 10
        font = None
        for x in xrange(self.visibletop,maximumpos):
            i,cb,args = items[x]
            #print i      
            if x == self._selected_index:                 
                del font # Close the font file
                font = self.sysdata.get_font_normal_b()
                color = ITEM_SELECTED_TEXT
                bgcolor = ITEM_SELECTED
            else:
                del font # Close the font file
                font = self.sysdata.get_font_normal()
                color = ITEM_UNSELECTED_TEXT
                bgcolor = ITEM_UNSELECTED
            
            s = ( size[0]-startposx*2- 15, spaceheight )
            pos  = ( startposx, startposy )
                        
            # Draw text
            text = font.render(i, 1, color)
            textpos = text.get_rect()
            textpos.centerx = self.parent.get_rect().centerx
            textpos.centery = pos[1] + s[1] / 2
            
            # Add to list            
            surf.blit( text, textpos )
            startposy = startposy + height
        
        self.items_changed = False
        
    
    def update_title(self):
        """ Update title surface """
        if not self.title_changed: return
        
        self.text = text = self.sysdata.get_font_title().render(self._title, 1, (0, 255, 0))
        self.textpos = textpos = text.get_rect()
        textpos.centerx = self.parent.get_rect().centerx
        textpos.centery = textpos.size[1] / 2 + 7
        
        self.size = size = self.parent.get_size()
        size = ( size[0], 40 )

        surf = titlebg = pygame.Surface(size, SRCALPHA)
        self.titlebgrect = titlebgrect = pygame.Rect( 5, 5, size[0]-10, size[1]-10)
        pygame.draw.rect(surf, TITLE_BG, titlebgrect)
        pygame.draw.rect(surf, TITLE_STROKE, titlebgrect, 2)
        surf.blit(self.text, textpos )
        
        self.title_changed = False
        
        self.titlesurface = surf
        # Position on parent
        self.titlepos = (0,0)
        
    def update(self):                    
        self.update_title()
        self.update_items()
        
        self.parent.blit( self.titlesurface, self.titlepos )
        self.parent.blit( self.itemsurface,  self.itemspos )
        
class Application(object):
    def __init__(self):
        if os.name == "e32":
            size = pygame.display.list_modes()[0]
            self.screen = pygame.display.set_mode(size, SRCALPHA)
        else:
            self.screen = pygame.display.set_mode( DISPLAY_SIZE, SRCALPHA ) 
        
        self.sysdata = SystemData()
        self.sysdata.screen = self.screen
        
        self.background = Background(self.sysdata)
        
        items = [("Applications", self.mhApplications,()),
                 ("Settings",self.mhSettings,()), 
                 ("About",self.mhAbout,()),
                 ("Exit",self.mhExit,()), ]
        self._main_menu = Menu(self.background.background, self.sysdata, 
                        title = "pygame launcher",
                        items = items,
                        cancel_callback = ( self.mhExit, () )
                        )
        self.focused = self._main_menu
        self.sprites = pygame.sprite.OrderedUpdates()
        self.sprites.add( self.background )
        self.sprites.add( self.focused )
        self.running = True
        self.clock = pygame.time.Clock()
        
        self.app_to_run = None
        
        #: Updated by foreground event
        self.isForeground = True
        
    def initialize(self):
        pass
    
    def run(self):
        """ Main application loop """
        self.initialize()       
        
        self.sysdata.tics = pygame.time.get_ticks()
        
        eventhandler = self.handleEvent
        while self.running:
            
            for event in pygame.event.get():
                print event
                eventhandler(event)
            
            if self.isForeground:
                self.sprites.update()
                
                self.screen.blit(self.background.background, (0,0))
                
                pygame.display.flip()
                
                self.clock.tick(22)
                
            else:
                # Longer delay when in backround
                self.clock.tick(1)

            tics = pygame.time.get_ticks()
            self.sysdata.ticdiff = tics - self.sysdata.tics
            self.sysdata.tics = tics
            
        return self.app_to_run
        
    def mhLaunchApplication(self, app_path):
        """ Menu handler for application item """
        
        if app_path is None:
            # Restore pygame launcher menu
            self.focused.clear()
            self.sprites.remove(self.focused)
            self.sprites.add(self._main_menu)
            self.focused = self._main_menu
            return
        
        self.app_to_run = app_path
        self.running = 0
        
    def mhApplications(self):
        """ Menu handler for Applications item """
        
        join = os.path.join
        appdir = join( THISDIR, "..", "apps" )
        apps = glob( join( appdir, "*.py" ) )
        apps += glob( join( appdir, "*.pyc" ) )
        apps += glob( join( appdir, "*.pyo" ) )

        items = []
        for a in apps:
            name = os.path.basename(a)
            name = ".".join( name.split(".")[:-1])
            if len(name) == 0: continue
            
            i = ( name, self.mhLaunchApplication, (a,) )
            items.append(i)
            
        items.append( ("Back", self.mhLaunchApplication, (None,)) )
        self.sprites.remove(self.focused)
        self.focused = Menu(self.background.background, self.sysdata, 
                        title = "Applications",
                        items = items,
                        cancel_callback = ( self.mhLaunchApplication, (None,) )
                        )
        self.sprites.add(self.focused)
        
    def mhExit(self):
        """ Menu handler for exit item """
        self.running = 0
        
    def mhSettings(self):
        """ Menu handler for settings item """
        self.sprites.remove(self.focused)
        self.focused = TextField(self.background.background, self.sysdata,
                        exit_callback = self._exitAbout,
                        title = "Settings",
                        text = "Begone! Nothing to see here!",
                        mode = TextField.MODE_CENTER
                        )
        self.sprites.add(self.focused)
    
    def _exitAbout(self):
        """ About view's exit handler"""
        
        self.sprites.remove(self.focused)
        self.sprites.add(self._main_menu)
        self.focused = self._main_menu
        
    def mhAbout(self):
        """ Menu handler for about item """
        self._main_menu = self.focused
        text = """
        -= pygame launcher =-
        
        
        www.pygame.org
        www.launchpad.net/pys60community
        
        Author: Jussi Toivola
        
        """
        
        self.sprites.remove(self.focused)
        self.focused = TextField(self.background.background, self.sysdata,
                        exit_callback = self._exitAbout,
                        title = "About",
                        text = text,
                        mode = TextField.MODE_CENTER
                        )
        self.sprites.add(self.focused)
        
    def isExitEvent(self, event ):
        """ @return True if event causes exit """
        
        #if event.type == pygame.KEYDOWN:     
            # K_ESCAPE = Right softkey       
            #if event.key == pygame.constants.K_ESCAPE:
                #self.running = 0
                #return True
        if event.type == pygame.QUIT:
            return True
        
        return False
        
    def handleEvent(self, event ):
        if self.isExitEvent(event):
            print "Exit event received!"
            self.running = 0
            return
        
        if self.isForeground:
            if self.focused is not None:
                handled = self.focused.handleEvent(event)
                if not handled:
                    # K_ESCAPE = Right softkey       
                    if event.type == pygame.KEYDOWN \
                    and event.key == constants.K_ESCAPE:
                        self.running=0
    
def start():
    """ Start pygame launcher """
         
    pygame.init() 
    while True:
                            
        a = Application()
        # The executable is received
        path_to_app = a.run()
                  
        # Clear cyclic references and the launcher out of the way        
        del a.background.sysdata
        del a.background
        del a._main_menu.sysdata
        del a._main_menu.parent
        del a._main_menu._items
        del a._main_menu
        del a.focused
        del a.sysdata
        a.sprites.empty()
        del a
        
        if path_to_app:
            
            # Run the application and restart launcher after app is completed.
            try:
                os.chdir(os.path.dirname(path_to_app))
                execfile(path_to_app, {'__builtins__': __builtins__,
                                       '__name__': '__main__',
                                       '__file__': path_to_app,
                                       'pygame' : pygame }
                )
            except:
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                        
            
        else:
            # Exit launcher
            break
        
    pygame.quit()
    
if __name__ == "__main__":
    start()
