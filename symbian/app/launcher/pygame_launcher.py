"""
pygame launcher for S60 - Application for launching others
"""
import time
__author__ = "Jussi Toivola"

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
    
        # Cache fonts to improve performance
        # Fonts using the same font file works with OpenC only.
        # SDL wants to keep the font's file handle open.
        # Symbian's c-library (estlib) does not let one to have
        # multiple handles open to a single file.
        # (even though the Symbian's RFile implementation does )
        self.font_title  = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 18)
        self.font_normal = pygame.font.Font(None, 25)
        
        self.font_normal_bold = pygame.font.Font(None, 25)
        self.font_normal_bold.set_bold(True)
        
    # Used by TextCache class, which caches the surfaces of the rendered texts.
    def getFontTitle(self):
        return self.font_title
    def getFontNormal(self):
        return self.font_normal
    def getFontNormalBold(self):
        return self.font_normal_bold
    def getFontSmall(self):
        return self.font_small

class Effects(object):
    """ Class for generic effects"""
    
    def __init__(self, screen, clock, surf1, surf2, render_callback = None):
        self.screen = screen
        self.clock = clock
        self.surf1 = surf1
        self.surf2 = surf2 
        
        self.duration = 0
        self.tween    = lambda x:None
        
        #: Called for each update to allow update of other surfaces.
        self.render_callback = render_callback
    
    def tween_easeout(self, x):
        return 1 - (1 - x) * (1 - x);
    
    def tween_easeinout(self, x):
        return ((x) * (x) * (3 - 2. * (x)))
    
    def _slide_and_replace(self, direction):
        
        # See http://sol.gfxile.net/interpolation/index.html about tween animations
        r = self.surf1.get_rect()
        
        tween = self.tween
        
        tics  = 20
        steps = 15.
        A = 0
        B = -r.width
        surfold = self.surf1
        surfnew = self.surf2

        # Time per step
        #for i in xrange(int(steps)):
        start = time.time()
        end   = start + self.duration
        while time.time() < end:
            v = ( time.time() - start ) / self.duration
            #v = i / steps;
            v = tween(v);
            X = (A * v) + (B * (1 - v));
            #X = direction * X
            #print direction,X
            
            if direction == -1: 
                self.screen.blit( surfold, (B-X, 0 ) )
                self.screen.blit( surfnew, ( -X, 0 ) )
            elif direction == 1:
                self.screen.blit( surfold, ((-B)+X, 0 ) )
                self.screen.blit( surfnew, (X, 0 ) )
            
            pygame.display.flip()
            self.clock.tick(tics)
            
            if self.render_callback is not None:
                self.render_callback()
            
    def effectSlideRightReplace(self):
        """ Effect makes the old surface move right and the new slides in it's place """
        return self._slide_and_replace(1)
    
    def effectSlideLeftReplace(self):
        """ Effect makes the old surface move left and the new slides in it's place """
        return self._slide_and_replace(-1)
        
    def do(self, effect, tween, duration):
        """
        @param effect: Function of the main effect.
        @param tween:  Function of transition effect.
        @param duration: How long the effect takes in seconds( 1, 0.5, etc )
        """
        self.duration = duration
        self.tween    = tween
        return effect( )
        
class TextCache(object):
    """ Handles text rendering and caches the surfaces of the texts for speed.
    Suitable for small static texts, such as menu items and titles.
    To avoid memory problems with very long lists, maximum cache size can be set.
    If the cache's size is exceeded the new surface is added into the cache and
    the oldest is removed.
    """
    def __init__(self, max_size=0):
        
        #: id->surface mapping
        self.map = {}
        
        self.max_size = max_size
        
        #: dict does not preseve order so we'll need to manage it with a list
        self.order = []
        
    def render(self, id, string, fontgetter, renderargs, force_update = False ):
        """ Render string on the first time, but use it's cached surface next time. 
        @param id: ID of the string
        @param string: The text to render.
        @param fontgetter: Function taking no parameters which returns a Font object.
        @param renderargs: Arguments for font.render
        @param force_update: Force re-rendering of the surface.
        """
        
        # Check if exists
        if id in self.map and not force_update:
            return self.map[id]
        
        font = fontgetter()
        surface = font.render(string, *renderargs)
        self.map[id] = surface
        
        # Update cache's max size.
        if self.max_size != 0:
            
            # No need to handle order if max_size is not set
            self.order.append(id)
            if len(self.order) > self.max_size:
                del self.map[self.order[0]]
                del self.order[0]
        
        return surface

class DrawUtils:
    """ Utility class for drawing common UI components. """
    
    @classmethod
    def drawRectWithText(cl, surf, size, bgcolor, fgcolor, textsurf = None, textpos = None):
        """
        @param surf: Surface to draw to
        @param size: Size of the rect
        @param bgcolor: Background color of the rect
        @param fgcolor: Foreground color of the rect( Text and surrounding rect )
        @param textsurf: Surface containing pre-rendered text
        @param textpos: Position of the text surface on the 'surf'
        @param startpos: Start position
        """
        
        # Make the dimmer( alpha ) foreground color for faded border
        dim = list(fgcolor)
        dim[-1] *= 0.5
        
        if len(size) == 2:
            rect = pygame.Rect( 4, 4, size[0]-7, size[1]-7)
        else:
            rect = pygame.Rect( *size )
        
        # Draw the background
        pygame.draw.rect(surf, bgcolor, rect)
        
        # Blit the text surface if defined
        if textsurf is not None: 
            surf.blit(textsurf, textpos )
        
        # Draw dim outer rect
        pygame.draw.rect(surf, dim, rect, 1)
        
        # Draw the center rect
        alpha_diff = abs( fgcolor[-1] - bgcolor[-1])
        
        for x in xrange(0,3):
            # Draw dim inner rect
            color = list(fgcolor)
            color[-1] -= ( alpha_diff / 3. * x )
            offset = rect[0] + x

            pos = pygame.Rect( rect[0] + x, rect[1] + x, rect[2]-(x*2-1), rect[3]-(x*2-1))
            #print rect, pos
            pygame.draw.rect(surf, color, pos, 1)
            
        
class Background(pygame.sprite.Sprite):
    """Main background"""
    def __init__(self, sysdata):
        pygame.sprite.Sprite.__init__(self) 
        
        self.sysdata = sysdata
        
        self.screen = sysdata.screen
        screen_size = self.screen.get_size()
        # TODO: Rename to 'surface'
        self.background = pygame.Surface(screen_size, SRCALPHA)
        self.background.fill(BLACK)
        
        self.rect = self.background.get_rect()
        middle    = self.rect.center
        
        self.img_logo, r = load_image( "logo.jpg")
        self.img_pos = [ middle[0] - r.w / 2, middle[1] - r.h / 2 ]
        # Make alpha the same size as the image
        self.alpha = pygame.Surface(r.size, SRCALPHA)
        
        self.alphaval = 200.
        self.alphadir = -20. # per second
         
    def updateAlphaValue(self):
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
        
        self.updateAlphaValue()
        
        self.background.fill(BLACK)
        
        self.alpha.fill( (0,0,0,self.alphaval) )
        
        self.background.blit( self.img_logo, self.img_pos )
        self.background.blit( self.alpha, self.img_pos )
        

class TextField(pygame.sprite.Sprite):
    """ Handles text rendering and updating when necessary """
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
        
    def updateText(self):
        """ Redraw text contents """
        if not self.text_changed: return
        
        text = self._text
        
        # Below title
        startposy = self.titlesurface.get_size()[1] + 10
        startposx = 10
        
        # Position on parent
        self.itemspos = ( 0, startposy )
        
        psize = self.parent.get_size()
        size = ( psize[0], psize[1] - startposy )
        surf = pygame.Surface(size, SRCALPHA)
        
        # Create text contents
        DrawUtils.drawRectWithText( surf, size, TITLE_BG, TITLE_STROKE, 
                                    textsurf = None )
        #text = textwrap.dedent(text)
        font = self.sysdata.getFontSmall()
        
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
        
    def updateTitle(self):
        """ Redraw title text """
        if not self.title_changed: return
            
        text = self.sysdata.getFontTitle().render(self._title, 1, (0, 255, 0))
        textpos = text.get_rect()
        textpos.centerx = self.parent.get_rect().centerx
        textpos.centery = textpos.size[1] / 2 + 7
        
        self.size = size = self.parent.get_size()
        size = ( size[0], 40 )

        # Draw the surrounding rect
        surf = titlebg = pygame.Surface(size, SRCALPHA)
        DrawUtils.drawRectWithText( surf, size, TITLE_BG, TITLE_STROKE, 
                                    textsurf = text, textpos = textpos )
        
        self.title_changed = False
        self.titlesurface = surf
        # Position on parent
        self.titlepos = (0,0)
        
    def update(self):
        self.updateTitle()
        self.updateText()
        
        self.parent.blit(self.titlesurface, self.titlepos )
        self.parent.blit(self.textsurface, self.itemspos )
    
    def exit(self):
        self.exit_callback()
        
    def handleEvent(self, event ):
        """ Exit on any key """
        if event.type == pygame.KEYDOWN:
            self.exit()
            return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.exit()
            return True
        
        return False
    
class Menu(pygame.sprite.Sprite):
    
    def __init__(self, bg, sysdata, title, items, cancel_callback):
        pygame.sprite.Sprite.__init__(self)
        
        #: General information about the system
        self.sysdata = sysdata
        
        #: Background object
        self.bg = bg
        
        #: Parent surface where the menu contents are blit on.
        self.parent = bg.background
        
        #: Text of the title
        self._title = title
        
        #: If True, title surface is updated
        self.title_changed = True
        
        #: Surface containing the rendered menu items.
        self.itemsurface = None
        
        #: Strings of the items
        self._items = items
        
        #: Rects of list items to be used for mouse hit checking
        self._itemrects = []
        
        #: If True, the item surfaces are updated
        self.items_changed = True
        
        #: Index of selected menu item
        self._selected_index = 0
        
        #: Index of previous selection
        self._prev_index = 0
        
        #: Index of the topmost visible item
        self.visibletop = 0
        
        #: How many items the list can display
        self.shownitems = 0
        
        #: Callback called at exit
        self.cancel_callback = cancel_callback
        
        #: Cached texts
        self.textcache = TextCache()
        
        #: Flag to indicate if mouse/pen is down
        self._mousedown = False
        
        #: Rect for scrollbar. Used to scroll the list with mouse/pen.
        self._scrollbar_rect = Rect(0,0,0,0)
        
        #: Rect for scrollbar indicator. User can drag this around with mouse/pen.
        self._scrollbar_indic_rect = Rect(0,0,0,0) 

    #------------------------------------------------ Selection property get/set
    def _set_selection(self,index):
        self._prev_index = self._selected_index
        self._selected_index = index
        self.items_changed = True
        self.updateVisible()
        
    def _get_selection(self): return self._selected_index
    selection = property(fget=_get_selection, fset=_set_selection )

    #---------------------------------------------------- Title property get/set
    def _set_title(self, title):
        self._title = title
        self.title_changed = True
        
    def _get_title(self): return self._title
    title = property(fget=_get_title, fset=_set_title)

    #----------------------------------------------------- Item property get/set
    def _set_items(self, title): 
        self._items = items
        self.items_changed = True
        
    def _get_items(self): return self._items
    items = property(fget=_get_items, fset=_set_items)

    #---------------------------------------------------------- Public functions
    def selectNextItem(self):
        """ Select next item from list. Loops the items """
        last = len(self._items) - 1
        if last < 1: return
        
        next = self._selected_index + 1
        if next > last:
            next = 0
        self.selection = next
        
    def selectPrevItem(self):
        """ Select previous item from list. Loops the items """
        last = len(self._items) - 1
        if last < 1: return
        
        first = 0
        next = self._selected_index - 1
        if next < first:
            next = last
        self.selection = next
         
    def doSelect(self):
        """ Handle item selection by invoking its callback function """
        title, callback,args = self._items[self._selected_index]
        callback(*args)
    
    def cancel(self):
        """ Invokes the menu cancellation handler """
        cb,args = self.cancel_callback
        cb(*args)
        
    def clear(self):
        " Remove the used surfaces from memory "
        self.itemsurface = None
        self.titlesurface = None
        self.items_changed = True
        self.title_changed = True
        
    def checkMouseCollision(self,event):
        """ Checks if mouse event collides with any of the menu items """
        # Not yet known.
        if self.itemsurface is None: return False
        
        # Check if we hit the surface containing the items
        menurect = pygame.Rect(self.itemsurface.get_rect())
        menurect.top = self.itemspos[1]
        
        if menurect.collidepoint(event.pos):
            r = Rect(self._scrollbar_rect)
            r.top += self.itemspos[1]
            if r.collidepoint(event.pos):
                print "scrollbar"
                if event.type in [pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN]:
                    r = Rect(self._scrollbar_indic_rect)
                    r.top += self.itemspos[1]
                    #self.visibletop = len(self.items) - 1
                    #self.items_changed = True
                    if r.collidepoint(event.pos):
                        # TODO: Need to do something to get rid of current selection being mandatory
                        # TODO: Handle indicator dragging
                        
                        print "Scrollbar indicator!"
                        return 2
                    else:
                        print "Scrollbar!"
                        # TODO: Handle user clicking empty scrollbar area to move the view area
                        return 2
            else:
                for x in xrange(len(self._itemrects)):
                    r = Rect(self._itemrects[x])
                    r.top += self.itemspos[1]
                    if r.collidepoint(event.pos):
                        self.selection = x
                        return 1
        
        return 0
    
    def handleEvent(self, event ):
        """ Handle events of this component """
        
        if event.type == pygame.KEYDOWN:
            if event.key == constants.K_DOWN:
                self.selectNextItem()
                return True
            
            elif event.key == constants.K_UP:
                self.selectPrevItem()
                return True
            
            if event.key == constants.K_RETURN:
                self.doSelect()
                return True
            
            if event.key == constants.K_ESCAPE:
                self.cancel()
                return True
        
        # Mouse button down and movement only marks the item selected
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.checkMouseCollision(event)
            self._mousedown = True
        
        # When pen is pressed on surface, user can keep changing the
        # selection and activate the selection by releasing the pen.
        elif event.type == pygame.MOUSEMOTION:
           self.checkMouseCollision(event)
        
        # Mouse button up selects the item
        elif event.type == pygame.MOUSEBUTTONUP:
            self._mousedown = False
            
            # User can cancel selection by moving the pen outside of all items
            if self.checkMouseCollision(event) == 1:
                self.doSelect()
        
        return False
    
     
    def computeVisibleItemCount(self):
        """ Calculate the amount of items that fit inside the list """

        height = self.sysdata.getFontNormal().get_height()
        h = self.itemsurface.get_height()
        c = int(round((h / height))) - 1
        return c
    
    def update(self):
        self.updateTitle()
        self.updateItems()
        
        self.parent.blit( self.titlesurface, self.titlepos )
        self.parent.blit( self.itemsurface,  self.itemspos )
    
    def _create_list_bg(self, size):
        
        surf = pygame.Surface(size, SRCALPHA)
        DrawUtils.drawRectWithText( surf, size, TITLE_BG, TITLE_STROKE, textsurf = None )
        self.itemsurface = surf
        return surf
    
    def _get_list_size(self):
        psize = self.parent.get_size()
        size  = ( psize[0], psize[1] - self.itemspos[1])
        return size
    
    def _draw_scrollbar(self, surf):
        r = surf.get_rect()
        xs = r[2] - 10
        ys = r[1] + 10
        h  = r[3] - 20
        w  = 10
        r = Rect(xs,ys,w,h)
        
        DrawUtils.drawRectWithText(surf, r, TITLE_BG, TITLE_STROKE, textsurf = None)
        
        # Compute the size of the scrollbar's indicator bar
        # Visible items vs total items
        factor = (float(self.shownitems)/float( len(self.items) ) )
        indic_h = h - 4
        indic_h *= min( 1.0, factor )
        
        # Compute the position of the scrollbar's indicator bar
        # topmost item vs how many items
        empty_h = h - indic_h - 4
        factor = float(self.visibletop) / float(len(self.items) - self.shownitems)
        ys += 2
        ys += empty_h * factor
        indic_r = Rect(xs+2,ys,6,indic_h)
        DrawUtils.drawRectWithText(surf, indic_r, TITLE_STROKE, TITLE_STROKE, textsurf = None)
        
        self._scrollbar_rect = r 
        self._scrollbar_indic_rect = indic_r
        
    def updateItems(self):
        """ Update list item surface """
        if not self.items_changed: return
        
        items = self._items
        
        # Below title
        startposy = self.titlesurface.get_size()[1] + 10
        startposx = 10
        self.itemspos = ( 0, startposy )
        
        # Create and cache the list background
        size            = self._get_list_size()
        surf             = self._create_list_bg(size)
        self.shownitems  = self.computeVisibleItemCount()
        
        # Initialize data
        startposy       = 5
        self.visibletop = min(self.visibletop, self._selected_index)
        maximumpos      = min(len(items), self.visibletop + self.shownitems )
        height          = self.sysdata.getFontNormal().get_height()
        spaceheight     = height + 10
        
        # Refresh positions of items for mouse support
        self._itemrects = []
        
        
        # Draw the items
        for x in xrange(self.visibletop,maximumpos):
            textdata,cb,args = items[x]
            # Textcache id
            id = textdata
            if x == self._selected_index:
                font = self.sysdata.getFontNormalBold
                color = ITEM_SELECTED_TEXT
                bgcolor = ITEM_SELECTED
                id = "_"+id
            else:
                font = self.sysdata.getFontNormal
                color = ITEM_UNSELECTED_TEXT
                bgcolor = ITEM_UNSELECTED
                
            s = ( size[0]-startposx*2- 15, spaceheight )
            pos  = ( startposx, startposy )
            
            # Use the text cache for speed.
            text = self.textcache.render( id, textdata, font, ( 1, color))
            textpos = text.get_rect()
            textpos.centerx = self.parent.get_rect().centerx
            textpos.centery = pos[1] + s[1] / 2
            
            # Add to list
            surf.blit( text, textpos )
            startposy = startposy + height
            
            self._itemrects.append(textpos)
        
        self._draw_scrollbar(surf)
        
        self.items_changed = False
        
    def updateTitle(self):
        """ Update title surface """
        if not self.title_changed: return
        
        # Render the title text
        text = self.sysdata.getFontTitle().render(self._title, 1, (0, 255, 0))
        self.textpos = textpos = text.get_rect()
        textpos.centerx = self.parent.get_rect().centerx
        textpos.centery = textpos.size[1] / 2 + 7
        
        self.size = size = self.parent.get_size()
        size = ( size[0], 40 )

        # Render the final title surface with combined background and text 
        surf = pygame.Surface(size, SRCALPHA)
        DrawUtils.drawRectWithText( surf, size , TITLE_BG, TITLE_STROKE, text, textpos )
        
        # Update data
        self.title_changed = False
        self.titlesurface  = surf
        
        # Position on parent
        self.titlepos = (0,0)
    
    def updateVisible(self):
        """ Updates position of the topmost visible item in the list """
        diff = abs(self.visibletop - self._selected_index)
        if diff >= self.shownitems:
            self.visibletop = max(0,min( self._selected_index - self.shownitems+1, self._selected_index ))
         
class Application(object):
    """ Main application handler.
    Takes care of system initialization and main application states.
    """
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
        self._main_menu = Menu(self.background, self.sysdata, 
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
        self.is_foreground = True
        
    def initialize(self):
        pass
    
    def run(self):
        """ Main application loop """
        self.initialize()       
        
        self.sysdata.tics = pygame.time.get_ticks()
        
        eventhandler = self.handleEvent
        while self.running:
            
            for event in pygame.event.get():
                #print event
                eventhandler(event)
            
            #print "update", self.is_foreground
            if self.is_foreground:
                self.sprites.update()
            
            self.screen.blit(self.background.background, (0,0))
            
            pygame.display.flip()
            
            if self.is_foreground:
                self.clock.tick(20)
            else:
                # Longer delay when in backround
                self.clock.tick(5)

            tics = pygame.time.get_ticks()
            self.sysdata.ticdiff = tics - self.sysdata.tics
            self.sysdata.tics = tics
            
        return self.app_to_run
        
    def mhLaunchApplication(self, app_path):
        """ Menu handler for application item """
        
        if app_path is None:
            # Restore pygame launcher menu
            
            self.__handle_transition_animation(self.focused, self._main_menu, dir = 1 )
            
            self.focused.clear()
            self.sprites.remove(self.focused)
            self.sprites.add(self._main_menu)
            self.focused = self._main_menu
            return
        
        self.app_to_run = app_path
        self.running = 0
    
    def __handle_transition_animation(self, menu1, menu2, dir):
        bg1 = menu1.bg
        bg2 = menu2.bg
        def render_callback():
            # We'll need to update the logo in the background, 
            # which depends on the tick counter.
            tics = pygame.time.get_ticks()
            self.sysdata.ticdiff = tics - self.sysdata.tics
            self.sysdata.tics = tics
            
            bg2.update()
            menu2.update()
        
        # Start the tween animation
        e = Effects(self.screen, self.clock, bg1.background, bg2.background,
                    render_callback = render_callback)
        
        # Blocks for the duration of the animation
        effect = [e.effectSlideLeftReplace,e.effectSlideRightReplace][dir]
        e.do(effect, e.tween_easeout, 0.5)
        
        # The animation completed. Set them for event handling.
        menu2.parent = self.background.background
        menu2.update()
        
        # Copy the alpha values to make the transition seem... smooth
        bg1.alphaval = bg2.alphaval
        bg1.alphadir = bg2.alphadir
        bg1.background.blit(bg2.background, (0,0))
        
    def mhApplications(self):
        """ Menu handler for 'Applications' item """
        
        # Get list of applications
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
        
        # The last button for getting out of the menu
        items.append( ("Back", self.mhLaunchApplication, (None,)) )
        
        self.sprites.remove(self.focused)
        
        # Create tween effect for transition
        background = Background(self.sysdata)
        
        menu = Menu(background, self.sysdata, 
                        title = "Applications",
                        items = items,
                        cancel_callback = ( self.mhLaunchApplication, (None,) ),
                        )
        menu.textcache.max_size = 24
        menu.update()
        self.__handle_transition_animation(self.focused, menu, dir = 0 )
        
        self.focused = menu
        self.sprites.add(self.focused)
        
    def mhExit(self):
        """ Menu handler for exit item """
        self.running = 0
    
    def mhSettings(self):
        """ Menu handler for settings item """
        self.sprites.remove(self.focused)
        self.focused = TextField(self.background.background, self.sysdata,
                        exit_callback = self._exit_to_main,
                        title = "Settings",
                        text = "Begone! Nothing to see here!",
                        mode = TextField.MODE_CENTER
                        )
        self.sprites.add(self.focused)
    
    def _exit_to_main(self):
        """ Callback to exit back to main menu """
        
        self.sprites.remove(self.focused)
        self.sprites.add(self._main_menu)
        self.focused = self._main_menu
        
    def mhAbout(self):
        """ Menu handler for about item """
        self._main_menu = self.focused
        text = """
        -= pygame launcher =-
        
        http://www.pygame.org
        
        http://code.google.com/p/
        pygame-symbian-s60/
        
        """
        
        self.sprites.remove(self.focused)
        self.focused = TextField(self.background.background, self.sysdata,
                        exit_callback = self._exit_to_main,
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
            #print "Exit event received!"
            self.running = 0
            return
        
        # Update foreground state
        if event.type == pygame.ACTIVEEVENT and os.name == "e32":
            self.is_foreground = event.gain
        
        if self.is_foreground:
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
        
        # Don't handle events given for launched application
        pygame.event.clear()
        
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
