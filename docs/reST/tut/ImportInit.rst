
```
import pygame
import random

# Initialize Pygame
pygame.init()

# Set the screen size
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Shape Shift Game")

# Set the colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

# Set the font
font_style = pygame.font.SysFont(None, 30)

# Define the function to display the message
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [screen_width / 6, screen_height / 6])

# Define the function to generate a random shape
def generate_shape():
    shape_list = ["square", "circle", "triangle"]
    return random.choice(shape_list)

# Set the initial shape
current_shape = generate_shape()

# Set the timer and score variables
time_left = 10
clock = pygame.time.Clock()
start_ticks = pygame.time.get_ticks()
score = 0

# Set the level variables
level = 1
shapes_per_level = 5
shapes_remaining = shapes_per_level

# Set the power-up variables
power_up_active = False
power_up_time = 0

# Set the game over variable
game_over = False

# Set the game loop
while not game_over:

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if shape_rect.collidepoint(pos):
                # Increase the score and generate a new shape
                score += 1
                shapes_remaining -= 1
                current_shape = generate_shape()
                # Activate the power-up if enough shapes have been completed
                if shapes_remaining == 0:
                    power_up_active = True
                    power_up_time = pygame.time.get_ticks()
                    shapes_per_level += 1
                    shapes_remaining = shapes_per_level
                    level += 1

    # Fill the screen with black
    screen.fill(black)

    # Draw the shape
    if current_shape == "square":
        shape_rect = pygame.draw.rect(screen, white, [screen_width / 2 - 50, screen_height / 2 - 50, 100, 100])
    elif current_shape == "circle":
        shape_rect = pygame.draw.circle(screen, white, [screen_width / 2, screen_height / 2], 50)
    elif current_shape == "triangle":
        shape_rect = pygame.draw.polygon(screen, white, [(screen_width / 2 - 50, screen_height / 2 + 50), (screen_width / 2 + 50, screen_height / 2 + 50), (screen_width / 2, screen_height / 2 - 50)])

    # Draw the score and level
    score_text = font_style.render("Score: " + str(score), True, white)
    screen.blit(score_text, [10, 10])
    level_text = font_style.render("Level: " + str(level), True, white)
    screen.blit(level_text, [screen_width - 100, 10])

    # Draw the shapes remaining
    shapes_remaining_text = font_style.render("Shapes Remaining: " + str(shapes_remaining), True, white)
    screen.blit(shapes_remaining_text, [screen_width / 2 - 75, screen_height - 30])

    # Draw the power-up timer
    if power_up_active:
        power_up_time_left = 10 - (pygame.time.get_ticks() - power_up_time) / 1000
        power_up_text = font_style.render("Power-up: " + str(int(power_up_time_left)) + "s", True, green)
        screen.blit(power_up_text, [10, screen_height - 30])
        if power_up_time_left <= 0:
            power_up_active = False

    # Draw the timer
    seconds_passed = (pygame.time.get_ticks() - start_ticks) / 1000
    time_left = 10 - seconds_passed
    if time_left <= 0:
        time_left = 0
        game_over = True
    timer_text = font_style.render("Time Left: " + str(int(time_left)) + "s", True, red)
    screen.blit(timer_text, [screen_width / 2 - 50, 10])

    # Update the screen
    pygame.display.update()

    # Set the FPS
    clock.tick(60)

# Display the game over message and score
message("Game Over. Your score is " + str(score), red)
pygame.display.update()

# Wait for 2 seconds before quitting
pygame.time.wait(2000)

# Quit Pygame
pygame.quit()
