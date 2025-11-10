import pygame
import random

pygame.init()

# Screen size
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bicycle Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 150, 0)
DARK_RED = (180, 0, 0)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
ORANGE = (255, 140, 0)

# Bicycle settings
bike_width = 50
bike_height = 70
bike_speed = 8

# Obstacle settings
obstacle_width = 50
obstacle_height = 50
initial_obstacle_speed = 7
obstacle_spawn_distance = 80

# Fonts
font = pygame.font.SysFont(None, 35)
large_font = pygame.font.SysFont(None, 60)

# Clock
clock = pygame.time.Clock()


def draw_bike(x, y):
    # Bike frame (main body)
    pygame.draw.rect(screen, DARK_GRAY, (x + 15, y + 20, 20, 25), border_radius=3)

    # Seat
    pygame.draw.rect(screen, BLACK, (x + 12, y + 15, 26, 8), border_radius=4)

    # Handlebars
    pygame.draw.line(screen, DARK_GRAY, (x + 25, y + 20), (x + 25, y + 5), 4)
    pygame.draw.line(screen, DARK_GRAY, (x + 15, y + 5), (x + 35, y + 5), 4)
    pygame.draw.circle(screen, RED, (x + 15, y + 5), 3)
    pygame.draw.circle(screen, RED, (x + 35, y + 5), 3)

    # Front wheel
    pygame.draw.circle(screen, BLACK, (x + 15, y + 55), 12)
    pygame.draw.circle(screen, DARK_GRAY, (x + 15, y + 55), 8)
    pygame.draw.circle(screen, LIGHT_GRAY, (x + 15, y + 55), 4)

    # Back wheel
    pygame.draw.circle(screen, BLACK, (x + 35, y + 55), 12)
    pygame.draw.circle(screen, DARK_GRAY, (x + 35, y + 55), 8)
    pygame.draw.circle(screen, LIGHT_GRAY, (x + 35, y + 55), 4)

    # Pedals
    pygame.draw.circle(screen, YELLOW, (x + 25, y + 40), 4)

    # Rider (simple stick figure)
    pygame.draw.circle(screen, (255, 200, 150), (x + 25, y + 8), 6)  # Head
    pygame.draw.line(screen, BLUE, (x + 25, y + 14), (x + 25, y + 28), 3)  # Body


def draw_obstacles(obstacles):
    for obs in obstacles:
        # Main box
        pygame.draw.rect(screen, RED, obs)
        # Top highlight
        pygame.draw.rect(screen, (255, 100, 100), (obs.x, obs.y, obs.width, 10))
        # Side shadow
        pygame.draw.rect(screen, DARK_RED, (obs.x + obs.width - 10, obs.y, 10, obs.height))
        # Danger stripes
        for i in range(0, obs.height, 15):
            pygame.draw.rect(screen, YELLOW, (obs.x + 5, obs.y + i, obs.width - 10, 5))


def show_score(score):
    text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(text, (10, 10))


def draw_button(x, y, width, height, text, color, hover_color, mouse_pos):
    button_rect = pygame.Rect(x, y, width, height)
    # Check if mouse is hovering
    is_hover = button_rect.collidepoint(mouse_pos)
    current_color = hover_color if is_hover else color

    # Draw button shadow
    shadow_rect = pygame.Rect(x + 4, y + 4, width, height)
    pygame.draw.rect(screen, DARK_GRAY, shadow_rect, border_radius=10)

    # Draw button
    pygame.draw.rect(screen, current_color, button_rect, border_radius=10)
    pygame.draw.rect(screen, BLACK, button_rect, 3, border_radius=10)

    # Draw text
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)

    return button_rect


def show_high_score(high_score):
    text = font.render(f"Best: {high_score}", True, BLACK)
    screen.blit(text, (10, 50))


def game_over_screen(score, high_score, mouse_pos):
    screen.fill((240, 240, 255))

    # Draw road background
    for i in range(0, HEIGHT, 40):
        pygame.draw.rect(screen, LIGHT_GRAY, (WIDTH // 2 - 100, i, 200, 20))

    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(WHITE)
    screen.blit(overlay, (0, 0))

    # Game Over text with shadow
    game_over_text = large_font.render("GAME OVER!", True, BLACK)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2 + 3, HEIGHT // 2 - 83))
    game_over_text = large_font.render("GAME OVER!", True, RED)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 86))

    # Score
    score_text = font.render(f"Your Score: {score}", True, BLACK)
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 20))

    # High Score
    high_score_text = font.render(f"High Score: {high_score}", True, ORANGE)
    screen.blit(high_score_text, (WIDTH // 2 - high_score_text.get_width() // 2, HEIGHT // 2 + 20))

    # Draw buttons
    restart_button = draw_button(WIDTH // 2 - 100, HEIGHT // 2 + 70, 200, 50, "🔄 PLAY AGAIN", DARK_GREEN, GREEN, mouse_pos)
    quit_button = draw_button(WIDTH // 2 - 100, HEIGHT // 2 + 135, 200, 50, "❌ QUIT", DARK_RED, RED, mouse_pos)

    pygame.display.update()
    return restart_button, quit_button


def reset_game():
    bike_x = WIDTH // 2 - bike_width // 2
    bike_y = HEIGHT - bike_height - 10
    obstacles = []
    score = 0
    obstacle_speed = initial_obstacle_speed
    return bike_x, bike_y, obstacles, score, obstacle_speed


# Game variables
high_score = 0
bike_x, bike_y, obstacles, score, obstacle_speed = reset_game()

# Game states
PLAYING = 0
GAME_OVER = 1
game_state = PLAYING

# Main game loop
running = True
while running:
    if game_state == PLAYING:
        # Draw road background
        screen.fill((100, 100, 100))

        # Draw road lanes
        for i in range(0, HEIGHT, 60):
            y = (i + score // 2) % HEIGHT
            pygame.draw.rect(screen, YELLOW, (WIDTH // 2 - 5, y, 10, 30))

        # Draw road sides
        pygame.draw.rect(screen, GREEN, (0, 0, 50, HEIGHT))
        pygame.draw.rect(screen, GREEN, (WIDTH - 50, 0, 50, HEIGHT))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and bike_x > 0:
            bike_x -= bike_speed
        if keys[pygame.K_RIGHT] and bike_x < WIDTH - bike_width:
            bike_x += bike_speed

        # Add new obstacles
        if len(obstacles) == 0 or obstacles[-1].y > obstacle_spawn_distance:
            obstacle_x = random.randint(50, WIDTH - obstacle_width - 50)
            obstacles.append(pygame.Rect(obstacle_x, -obstacle_height, obstacle_width, obstacle_height))

        # Increase difficulty
        if score > 0 and score % 300 == 0:
            obstacle_speed += 0.3
            obstacle_spawn_distance = max(50, obstacle_spawn_distance - 2)

        # Move obstacles
        for obs in obstacles:
            obs.y += obstacle_speed

        # Check collisions
        bike_rect = pygame.Rect(bike_x, bike_y, bike_width, bike_height)
        for obs in obstacles:
            if obs.colliderect(bike_rect):
                game_state = GAME_OVER
                if score > high_score:
                    high_score = score

        # Remove off-screen obstacles and update score
        old_count = len(obstacles)
        obstacles = [obs for obs in obstacles if obs.y < HEIGHT]
        if len(obstacles) < old_count:
            score += 10

        # Draw everything
        draw_bike(bike_x, bike_y)
        draw_obstacles(obstacles)
        show_score(score)
        show_high_score(high_score)

        pygame.display.update()
        clock.tick(30)

    elif game_state == GAME_OVER:
        mouse_pos = pygame.mouse.get_pos()
        restart_button, quit_button = game_over_screen(score, high_score, mouse_pos)

        # Wait for restart or quit
        waiting = True
        while waiting:
            mouse_pos = pygame.mouse.get_pos()
            restart_button, quit_button = game_over_screen(score, high_score, mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button.collidepoint(mouse_pos):
                        # Restart game
                        bike_x, bike_y, obstacles, score, obstacle_speed = reset_game()
                        game_state = PLAYING
                        waiting = False
                    elif quit_button.collidepoint(mouse_pos):
                        running = False
                        waiting = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Restart game
                        bike_x, bike_y, obstacles, score, obstacle_speed = reset_game()
                        game_state = PLAYING
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        waiting = False

pygame.quit()
