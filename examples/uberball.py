#/usr/bin/env python
"""
ÜberBall python source by Geoff Howland.  04-28-02

All rights to this code and source and name are placed in the public domain.

This is meant to be an example of how to use Python for creating games, and being my first project
in Python could have some things that could have been organized better, but I think it handles most
of the situations pretty elegantly.  The code was originally based on the 'chimp.py' example that
with PyGame examples as it was the most basic example to do full drawing/events.  Most, but not all,
of that code has been rewriten though so it probably wont be much to compare.

If you're interested in independent game development you can take a look at my website Ludum Dare:
http://ludumdare.com/

Hope this helps some people!

-Geoff Howland
ghowland@lupinegames.com


On May 21, 2002, this was slightly cleaned up to be included with the pygame
examples. Changes mainly include better use of the pygame.sprite module. All
graphic resources are now rendered during runtime instead of loaded from disk.
"""


#Import Modules
import os, pygame
from pygame.locals import *
from random import randint

if not pygame.font: raise SystemExit, "Requires pygame.font module"
if not pygame.mixer: print 'Warning, sound disabled'


#functions to create our resources

def load_sound(name):
	class NoneSound:
		def play(self): pass
	if not pygame.mixer or not pygame.mixer.get_init():
		return NoneSound()
	fullname = os.path.join('data', name)
	try:
		sound = pygame.mixer.Sound(fullname)
	except pygame.error, message:
		print 'Cannot load sound:', fullname
		raise SystemExit, message
	return sound

def render_block(size, color, width=1):
        hicolor = map(lambda x: x+40, color)
        locolor = map(lambda x: x-20, color)
        surf = pygame.Surface(size)
        r = surf.get_rect()
        smallr = r.inflate(-width-1, -width-1)
        surf.fill(color)
        pygame.draw.lines(surf, locolor, 0, (smallr.topright, smallr.bottomright, smallr.bottomleft), width)
        pygame.draw.lines(surf, hicolor, 0, (r.bottomleft, r.topleft, r.topright), width)
        return surf, r

def render_ball(radius, color):
        hicolor = map(lambda x: x+50, color)
        radius = radius
        size = radius * 2
        surf = pygame.Surface((size, size))
        pygame.draw.circle(surf, color, (radius, radius), radius)
        half = radius/2
        surf.set_at((half, half+1), hicolor)
        surf.set_at((half+1, half), hicolor)
        surf.set_colorkey(0)
        return surf, surf.get_rect()
        

def render_powerup(color):
        if not pygame.font:
            surf = pygame.Surface((30, 30))
            surf.fill(color)
            return surf, surf.get_rect()
        
        font = pygame.font.Font(None, 15)
        font.set_bold(1)
        surf = font.render(":-)", 0, color)
        surf2 = pygame.transform.rotate(surf, -90)
        r = surf2.get_rect()
        pygame.draw.rect(surf2, color, r, 1)
        return surf2, r


class PlayGameState:
	"Class for the game's play state variables and control functions."
	def __init__(self):
		self.menuMode = 0
		self.score = 0
		self.level = 1
		self.effectCurrent = 0
		self.effectDuration = 0

	def NewGame(self):
		self.menuMode = 2 # Game
		self.score = 0
		self.level = 1
		self.effectCurrent = 0
		self.effectDuration = 0
		
		# Load level

	def ScoreAdd(self, scoreAdd):
		self.score = self.score + scoreAdd

#classes for our game objects
class Paddle(pygame.sprite.Sprite):
	"""moves a clenched paddle on the screen, following the mouse"""
	def __init__(self):
		pygame.sprite.Sprite.__init__(self) #call Sprite initializer
		self.image, self.rect = render_block((50, 15), (200, 180, 120), 2)

	def update(self):
		"move the paddle based on the mouse position"
		pos = pygame.mouse.get_pos()	# This should really be passed the mouse position
		self.rect.midtop = pos
		self.rect.bottom = 440		# Lock Paddle at the bottom of the screen


#classes for our game objects
class Brick(pygame.sprite.Sprite):
	"""moves a clenched paddle on the screen, following the mouse"""
	colors = (200, 50, 50),  (50, 200, 50), (50, 50, 200), (200, 200, 50)
	def __init__(self, type, x, y):
		pygame.sprite.Sprite.__init__(self) #call Sprite initializer
		self.type = type
		self.image, self.rect = render_block((30, 10), Brick.colors[type])
		self.active = 1
		self.rect.left = (x * self.rect.width) + 5
		self.rect.top = y * self.rect.height
		self.posX = x
		self.posY = y

#classes for our game objects
class PowerUp(pygame.sprite.Sprite):
	"""moves a clenched paddle on the screen, following the mouse"""
	colors = (150, 50, 50), (50, 150, 50)
	def __init__(self, type, speed, x, y):
		pygame.sprite.Sprite.__init__(self) #call Sprite initializer
		self.type = type
		self.speed = speed
		self.image, self.rect = render_powerup(PowerUp.colors[type])
		self.rect.left = x
		self.rect.top = y

	def update(self):
		# Fall
		self.rect.top = self.rect.top + self.speed

	def collide (self, test):
		if self.rect.colliderect (test.rect):
			return 1

		return 0
			
class Ball(pygame.sprite.Sprite):
	"""moves a monkey critter across the screen. it can spin the
	   monkey when it is punched."""
	radii = 8, 4
	def __init__(self, size,x=300,y=200):
		pygame.sprite.Sprite.__init__(self) #call Sprite intializer
		self.size = size
		self.image, self.rect = render_ball(Ball.radii[size], (50, 200, 200))
		screen = pygame.display.get_surface()
		self.area = screen.get_rect()
		self.area.left = self.area.left - 5
		
		self.rect.centerx = x
		self.rect.centery = y
		self.moveX = 4 + (2 * size)
		self.moveY = 4 + (2 * size)
		self.lastRect = self.rect

	def changeType(self):
		size = not self.size
		self.size = size
		x = self.rect.centerx
		y = self.rect.centery
		self.image, self.rect = render_ball(Ball.radii[size], (50, 200, 200))
		self.rect.centerx = x
		self.rect.centery = y
		
		# Set the speed based on the size, and keep its direction
		if size:
			self.moveX = self.moveX * 2
			self.moveY = self.moveY * 2
		else:
			self.moveX = self.moveX / 2
			self.moveY = self.moveY / 2

	def update(self):
		self.lastRect = self.rect
		newpos = self.rect.move((self.moveX, self.moveY))
		if not self.area.contains(newpos):
			if self.rect.centerx < 0 or self.rect.centerx > self.area.right:
				self.moveX = -self.moveX
			if self.rect.centery < 0 or self.rect.centery > self.area.bottom:
				self.moveY = -self.moveY
			newpos = self.rect.move((self.moveX, self.moveY))

		self.rect = newpos

	def collidePaddle (self, test):
		horReverse = 0
		## Collision from right
		if self.lastRect.right < test.rect.left and self.lastRect.bottom > test.rect.top:
			self.rect.right = test.rect.left
			self.moveX = -self.moveX
			horReverse = 1
			
		## Collision from left			
		if self.lastRect.left > test.rect.right and self.lastRect.bottom > test.rect.top:
			self.rect.left = test.rect.right
			self.moveX = -self.moveX
			horReverse = 1

		## Collision from above			
		if not horReverse and self.moveY > 0:
			self.rect.bottom = test.rect.top
			# If we haven't done a horizontal reverse then do a gradient angle change
			if not horReverse:
				if self.rect.centerx < test.rect.centerx:
					self.moveX = ((self.rect.centerx - test.rect.centerx) / (test.rect.width / 2.0)) * 4.0
					if self.moveX >= -1:
						self.moveX = -1
				else:
					self.moveX = ((self.rect.centerx - test.rect.centerx) / (test.rect.width / 2.0)) * 4.0
					if self.moveX <= 1:
						self.moveX = 1
		
			self.moveY = -self.moveY

	def collide (self, test):
		horReverse = 0
		# Collision from right
		if self.lastRect.right < test.rect.left and self.lastRect.bottom > test.rect.top:
			self.rect.right = test.rect.left-1
			self.moveX = -self.moveX
			horReverse = 1
			
		# Collision from left			
		if self.lastRect.left > test.rect.right and self.lastRect.bottom > test.rect.top:
			self.rect.left = test.rect.right+1
			self.moveX = -self.moveX
			horReverse = 1

		# Collision from above			
		if not horReverse and self.moveY > 0:
			self.rect.bottom = test.rect.top
			self.moveY = -self.moveY

		# Collision from below			
		if not horReverse and self.moveY < 0:
			self.rect.top = test.rect.bottom
			self.moveY = -self.moveY


class GameMode:
	"""This handles all the game mode activities, playing, scoring, dying, drawing, etc."""
	def __init__(self, gameControlObj, screenRect):
		self.gameControl = gameControlObj	# Mode can change to another mode by itself
		
		# Create rect bonudaries for the screen
		self.screenBoundary = screenRect
		self.powerchance = 3
		
		# Create fonts
		self.fontScore = pygame.font.Font(None, 26)
		self.fontLives = pygame.font.Font(None, 26)
		
		# Create sounds
		self.bong_sound = load_sound('bong.wav')
		
		self.reset ()

	def reset(self):
		# Create and clear the gameState
		self.gameState = PlayGameState()
		
		# Create paddle for player
		self.paddle = Paddle()
		self.allsprites = pygame.sprite.RenderUpdates(self.paddle)

		# Create balls
		self.allballs = pygame.sprite.Group((Ball(1), Ball(0)))
		self.allsprites.add(self.allballs)

		# Create bricks
		self.allbricks = pygame.sprite.Group()
		for y in range(8):
			for x in range(21):
				type = randint(0,3)
				brick = Brick(type, x, y)
				brick.add((self.allbricks, self.allsprites))

		# Power ups
		self.allpowerups = pygame.sprite.Group()

	def eventHandle(self):
		if self.gameControl.gameEvent.keystate[K_ESCAPE]:
			self.gameControl.setMode (0)
		return 0
		
	def update(self):
		# Update all the sprite objects we've added to this comprehensive list
		self.allsprites.update()

		# If we're under the paddle
		bottom = self.screenBoundary.bottom
		for powerUp in self.allpowerups.sprites():
			if powerUp.rect.bottom > bottom:
				powerUp.kill()
		for ball in self.allballs.sprites():
			if ball.rect.bottom > bottom:
				ball.kill()

		# Check balls against blocks
		collisiondict = pygame.sprite.groupcollide(self.allballs, self.allbricks, 0, 1)
		for ball,bricks in collisiondict.items():
			for brick in bricks:
				ball.collide(brick)
				self.bong_sound.play()
				self.gameState.ScoreAdd (10)
				# Randomly create a power up
				if not randint(0, self.powerchance):
					self.powerchance = len(self.allballs) * 2
					center = brick.rect.center
					powerUp = PowerUp(randint(0,1), randint(3,7), center[0], center[1])
					powerUp.add((self.allpowerups, self.allsprites))
				else:
					self.powerchance = self.powerchance - 1

		# Check balls against paddle
		for ball in pygame.sprite.spritecollide(self.paddle, self.allballs, 0):
			ball.collidePaddle(self.paddle)
			self.bong_sound.play()
			self.gameState.ScoreAdd(10)

		# Check powerups against paddle
		for powerUp in pygame.sprite.spritecollide(self.paddle, self.allpowerups, 1):
			ball.collidePaddle(self.paddle)
			self.bong_sound.play()
			if powerUp.type == 1: # Double the balls
				for ball in self.allballs.sprites():
					newBall = Ball (ball.size, ball.rect.centerx, ball.rect.centery)
					newBall.moveX = -ball.moveX
					newBall.moveY = -ball.moveY
					newBall.add((self.allballs, self.allsprites))
			else: # Convert the balls
				for ball in self.allballs.sprites():
					ball.changeType()

		# If all the balls are gone
		if not self.allballs:
			self.gameControl.setMode(0)	# Back to the main menu

	def draw(self, background, screen):
		#Draw Everything
		screen.blit(background, (0, 0))
		self.allsprites.draw(screen)
		
		# Draw score
		textScore = self.fontScore.render(str(self.gameState.score), 1, (255, 255, 255))
		textposScore = textScore.get_rect()
		textposScore.bottom = background.get_rect().bottom - 5
		textposScore.left = background.get_rect().left + 10
		screen.blit(textScore, textposScore)

		# Draw balls
		textLives = self.fontLives.render(str(len(self.allballs)), 1, (255, 255, 255))
		textposLives = textLives.get_rect()
		textposLives.bottom = background.get_rect().bottom - 5
		textposLives.right = background.get_rect().right - 10
		screen.blit(textLives, textposLives)

class MainMenuMode:
	"""This handles all the main menu activities of quitting, or starting a game, checking high score"""
	def __init__(self, gameControlObj, screenRect):
		self.gameControl = gameControlObj	# Mode can change to another mode by itself
		self.screenBoundary = screenRect
		self.menuSelect = 0
		self.menuMax = 3

		# Create fonts
		self.fontMenu = pygame.font.Font(None, 30)

	def reset(self):
		return

	def eventHandle(self):
		# Check for quit
		if self.gameControl.gameEvent.keystate[K_ESCAPE]:
			self.gameControl.setMode (-1)	# -1 is exit the game
	
		# Move selection up and down
		if self.gameControl.gameEvent.newkeys[K_DOWN] and self.menuSelect < self.menuMax-1:
			self.menuSelect = self.menuSelect + 1
			
		if self.gameControl.gameEvent.newkeys[K_UP] and self.menuSelect > 0:
			self.menuSelect = self.menuSelect - 1
		
		# Process current selection
		if self.gameControl.gameEvent.newkeys[K_RETURN]:
			if self.menuSelect == 0:
				self.gameControl.setMode (2)
			if self.menuSelect == 1:
				self.gameControl.setMode (1)
			if self.menuSelect == 2:
				self.gameControl.setMode (-1)	# -1 is exit the game
		
	def update(self):
		return

	def draw(self, background, screen):
		#Draw Everything
		screen.blit(background, (0, 0))
		
		# Draw options - New Game
		color = (255, 255, 255)
		if self.menuSelect == 0:
			color = (0, 255, 0)
		textMenu = self.fontMenu.render("New Game", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.bottom = background.get_rect().centery - textPosMenu.height
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)
		lastBottom = textPosMenu.bottom

		# Draw options - High Score
		color = (255, 255, 255)
		if self.menuSelect == 1:
			color = (0, 255, 0)
		textMenu = self.fontMenu.render("High Scores", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.top = lastBottom+10
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)
		lastBottom = textPosMenu.bottom

		# Draw options - Quit
		color = (255, 255, 255)
		if self.menuSelect == 2:
			color = (0, 255, 0)
		textMenu = self.fontMenu.render("Quit", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.top = lastBottom+10
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)

class HighScoreMenuMode:
	"""This handles all the main menu activities of quitting, or starting a game, checking high score"""
	def __init__(self, gameControlObj, screenRect):
		self.gameControl = gameControlObj	# Mode can change to another mode by itself
		self.screenBoundary = screenRect

		# Create fonts
		self.fontMenu = pygame.font.Font(None, 30)

	def reset(self):
		return

	def eventHandle(self):
		# Quit on any keys
		if self.gameControl.gameEvent.keystate[K_RETURN]:
			self.gameControl.setMode (0)
		if self.gameControl.gameEvent.keystate[K_SPACE]:
			self.gameControl.setMode (0)
		
	def update(self):
		return

	def draw(self, background, screen):
		#Draw Everything
		screen.blit(background, (0, 0))
		
		# This could be more dynamic and function, but I dont really feel like it right now.
		
		# Draw options - New Game
		color = (255, 255, 255)
		textMenu = self.fontMenu.render("High Scores", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.bottom = background.get_rect().centery - textPosMenu.height
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)
		lastBottom = textPosMenu.bottom

		# Draw options - High Score
		color = (255, 255, 255)
		textMenu = self.fontMenu.render("----", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.top = lastBottom+10
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)
		lastBottom = textPosMenu.bottom

		# Draw options - Quit
		color = (255, 255, 255)
		textMenu = self.fontMenu.render("----", 1, color)
		textPosMenu = textMenu.get_rect()
		textPosMenu.top = lastBottom+10
		textPosMenu.left = background.get_rect().centerx - textPosMenu.width/2
		screen.blit(textMenu, textPosMenu)

class GameEvent:
	"""Event system wrapped so that it is based on time things were pressed.  
		Otherwise repeats occur that we dont desire."""
		
	def __init__(self):
		# Run update to init all the variables
		self.keystate = None
		self.update()

	def update(self):
		pygame.event.pump() #keep messages alive

		# Get the key state
		if self.keystate:
			oldstate = self.keystate
			self.keystate = pygame.key.get_pressed()
			self.newkeys = map(lambda o,c: c and not o, oldstate, self.keystate)
		else:
			self.keystate = pygame.key.get_pressed()
			self.newkeys = self.keystate

		# Get the mouse data
		self.mousePos = pygame.mouse.get_pos()
		self.mouseButtons = pygame.mouse.get_pressed()

	        # Get the time
	        self.ticks = pygame.time.get_ticks()

class GameControl:
	"""This saves the state that the game is in: what mode we're in, etc.  
		This is different than GameState because it deals with the play state."""
	def __init__(self):
		self.gameEvent = GameEvent()
		self.modeCur = 0
		self.modes = []
		return
		
	def addMode(self, newMode):
		"""Insert the new mode into the modes list"""
		self.modes.insert (len(self.modes), newMode)
		
	def setMode(self, newMode):
		"""Set the new mode, and reset it"""
		self.modeCur = newMode
		
		# If we didn't set the mode to exit
		if self.modeCur != -1:
			self.modes[self.modeCur].reset()		

	def update(self):
		"""Update the current mode and events"""
		self.gameEvent.update()
		
		if self.modeCur != -1:
			self.modes[self.modeCur].eventHandle()
			
		if self.modeCur != -1:
			self.modes[self.modeCur].update()

	def draw(self, background, screen):
		if self.modeCur != -1:
			self.modes[self.modeCur].draw(background, screen)


def main():
	"""this function is called when the program starts.
	   it initializes everything it needs, then runs in
	   a loop until the function returns."""
	   
	#Initialize Everything
	pygame.init()
	screen = pygame.display.set_mode((640, 480))
	pygame.display.set_caption('ÜberBall')
	pygame.mouse.set_visible(0)
	pygame.event.set_grab(1)
	
	#Create The Backgound
	background = pygame.Surface(screen.get_size())
	background = background.convert()
	background.fill((0, 0, 0))
	
	#Put Text On The Background, Centered
	if pygame.font:
		font = pygame.font.Font(None, 36)
		text = font.render("ÜberBall", 1, (255, 255, 255))
		textpos = text.get_rect()
		textpos.centerx = background.get_rect().centerx
		background.blit(text, textpos)
		lastBot = textpos.bottom
		
		font = pygame.font.Font(None, 20)
		text = font.render("by Geoff Howland", 1, (255, 255, 255))
		textpos = text.get_rect()
		textpos.centerx = background.get_rect().centerx
		textpos.top = lastBot + 10
		background.blit(text, textpos)


	# Create clock to lock framerate
	clock = pygame.time.Clock()

	# Create the game control object and add the game modes
	gameControl = GameControl()
	gameControl.addMode( MainMenuMode (gameControl, screen.get_rect()) )
	gameControl.addMode( HighScoreMenuMode (gameControl, screen.get_rect()) )
	gameControl.addMode( GameMode (gameControl, screen.get_rect()) )

	# Main Loop
	while 1:
		# Lock the framerate
		clock.tick(60)
	
		# Handle the modes
		gameControl.update()
		gameControl.draw(background, screen)

		# Handle game exit
		if gameControl.modeCur == -1:
			return

		# Flip to front
		pygame.display.flip()


#this calls the 'main' function when this script is executed
if __name__ == '__main__': main()
