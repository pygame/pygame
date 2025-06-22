import pygame
import random
import sys

# إعدادات اللعبة
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
BLOCK_SIZE = 20
FPS = 10

# الألوان
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# إعداد الشاشة
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("The Snake 🐍")
clock = pygame.time.Clock()

# الخطوط
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_small = pygame.font.SysFont("Arial", 24)

def draw_text_centered(text, font, color, y_offset=0):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        rendered = font.render(line, True, color)
        rect = rendered.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + y_offset + i*40))
        screen.blit(rendered, rect)

def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text_centered("The snake 🐍", font_large, RED, -50)
        draw_text_centered("Play", font_small, WHITE, 50)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                return

def game_loop():
    x = SCREEN_WIDTH // 2
    y = SCREEN_HEIGHT // 2
    dx = BLOCK_SIZE
    dy = 0

    snake = [(x, y)]
    length = 1

    # موقع أولي للتفاحة
    apple_x = random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    apple_y = random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

    score = 0

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and dx == 0:
            dx = -BLOCK_SIZE
            dy = 0
        elif keys[pygame.K_RIGHT] and dx == 0:
            dx = BLOCK_SIZE
            dy = 0
        elif keys[pygame.K_UP] and dy == 0:
            dx = 0
            dy = -BLOCK_SIZE
        elif keys[pygame.K_DOWN] and dy == 0:
            dx = 0
            dy = BLOCK_SIZE

        # تحديث الموقع
        x += dx
        y += dy

        # التحقق من الاصطدام بالجدران
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            running = False

        # تحديث جسم الثعبان
        snake.append((x, y))
        if len(snake) > length:
            del snake[0]

        # التحقق إذا أكلت التفاحة
        if x == apple_x and y == apple_y:
            score += 1
            length += 1
            apple_x = random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            apple_y = random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        # رسم العناصر
        screen.fill(BLACK)

        # رسم التفاحة (بيضاء)
        pygame.draw.circle(screen, WHITE, (apple_x + BLOCK_SIZE//2, apple_y + BLOCK_SIZE//2), BLOCK_SIZE//2)

        # رسم الثعبان (أحمر)
        for segment in snake:
            pygame.draw.rect(screen, RED, (*segment, BLOCK_SIZE, BLOCK_SIZE))

        # رسم السكور
        score_text = font_small.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

    # نهاية اللعبة
    game_over(score)

def game_over(score):
    while True:
        screen.fill(BLACK)
        draw_text_centered(f"Game Over\nScore: {score}\nPress any key to exit", font_small, WHITE)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                sys.exit()

# تشغيل اللعبة
main_menu()
game_loop()
