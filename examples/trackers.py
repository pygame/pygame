""" pygame.examples.trackers

A simple white-space field with four trackers that constantly home in on the mouse.

The trackers use velocity and acceleration to gravity around the mouse instead of move directly towards it.

Shows a more complicated way of programming in Pygame using Object-Oriented-Progamming.

The four default trackers behave differently:
* Red moves towards the player at a steady pace.
* Green is similar to red, but moves much faster and is slightly more percise at slower speeds.
* Blue has a high maximum velocity, but struggles to turn on a dime.
* Yellow has a low maximum velocity, but can easily turn on a dime.

Controls
--------

* Move the mouse to change the location the trackers are homing towards.
* Hold right click to freeze the trackers and reset their velocities.

"""

# Import basic modules
import pygame
import time

pygame.init()

# Game properties
screenWidth = 1000
screenHeight = 1000
tickDelay = 0.01  # Time between updates. Lower time = faster game.
gameTitle = "Tracker Field"

screen = pygame.display.set_mode([screenWidth, screenHeight])
pygame.display.set_caption(gameTitle)

class Ball:
    def __init__(self, x, y, colour, size, maxAcceleration, maxSpeed):
        self.position = [x, y]
        self.maxAcceleration = maxAcceleration
        self.maxSpeed = maxSpeed
        self.colour = colour
        self.size = size

        self.velocity = [0, 0]
    
    def updatePosition(self, mousePos):
        for i in range(0, 2):
            # Acceleration
            difference = mousePos[i] - self.position[i]
            difference = float(max(min(float(difference / 2), self.maxAcceleration), (-1 * self.maxAcceleration)))

            # Velocity
            self.velocity[i] += difference
            if self.velocity[i] > self.maxSpeed:
                self.velocity[i] = self.maxSpeed
            elif self.velocity[i] < self.maxSpeed * -1:
                self.velocity[i] = self.maxSpeed * -1
            
            # Displacement
            self.position[i] += self.velocity[i]

# Game class
class Game:
    def __init__(self):
        self.ballStats = [ # Starting X, Starting Y, Colour, Size, Max Acceleration, Max Velocity
            [200, 200, (255, 0, 0), 10, 0.1, 5],
            [800, 800, (0, 255, 0), 8, 0.3, 7],
            [200, 800, (0, 0, 255), 10, 0.05, 15],
            [800, 200, (255, 255, 0), 7, 0.2, 2]
        ]
        self.backgroundColour = (255, 255, 255)

        self.trackers = []
        for stat in self.ballStats:
            self.trackers.append(Ball(stat[0], stat[1], stat[2], stat[3], stat[4], stat[5]))

    def logic(self): # Logic function - Game logic is handled here.
        global screenWidth
        global screenHeight

        mousePos = pygame.mouse.get_pos()
        
        clicks = pygame.mouse.get_pressed()
        if clicks[2]: # Freeze all of the trackers and reset their velocities
            for ball in self.trackers:
                ball.velocity = [0, 0]
        else:
            for ball in self.trackers:
                ball.updatePosition(mousePos)

                # Loop trackers around
                if ball.position[0] < (ball.size * -1):
                    ball.position[0] = screenWidth + ball.size
                elif ball.position[0] > (ball.size + screenWidth):
                    ball.position[0] = -1 * ball.size
                
                if ball.position[1] < (ball.size * -1):
                    ball.position[1] = screenHeight + ball.size
                elif ball.position[1] > (ball.size + screenHeight):
                    ball.position[1] = -1 * ball.size

    def draw(self): # Draw function - Drawing the trackers is handled here.
        screen.fill(self.backgroundColour)
        for ball in self.trackers:
            pygame.draw.circle(screen, ball.colour, (int(ball.position[0]), int(ball.position[1])), ball.size)
        pygame.display.flip() # Display drawn objects on screen

# Run the actual game
game = Game() # Game object

gameRunning = True
while gameRunning:
    ev = pygame.event.get()

    for event in ev:
        if event.type == pygame.QUIT:
            gameRunning = False
    
    game.logic()
    game.draw()

    time.sleep(tickDelay)

pygame.quit()
