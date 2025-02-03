import pygame
import sys

# 初始化Pygame
pygame.init()

# 游戏常量
WIDTH = 800
HEIGHT = 600
GRAVITY = 0.8
JUMP_FORCE = -15
PLAYER_SPEED = 5

# 颜色定义
BLUE = (0, 120, 255)
BROWN = (165, 42, 42)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 60))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(center=(100, HEIGHT-100))
        
        # 芦泽的属性
        self.speed = PLAYER_SPEED
        self.jump_force = JUMP_FORCE
        self.y_velocity = 0
        self.on_ground = True
        self.score = 0
        self.lives = 3

    def update(self, platforms):
        keys = pygame.key.get_pressed()
        
        # 芦泽移动
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
            
        # 芦泽跳跃
        if keys[pygame.K_SPACE] and self.on_ground:
            self.y_velocity = self.jump_force
            self.on_ground = False

        # 重力应用
        self.y_velocity += GRAVITY
        self.rect.y += self.y_velocity

        # 平台碰撞检测
        self.on_ground = False
        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self.y_velocity > 0:
                    self.rect.bottom = platform.rect.top
                    self.on_ground = True
                    self.y_velocity = 0
                elif self.y_velocity < 0:
                    self.rect.top = platform.rect.bottom
                    self.y_velocity = 0

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(BROWN)
        self.rect = self.image.get_rect(topleft=(x, y))

class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect(center=(x, y))

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("芦泽冒险记")
    clock = pygame.time.Clock()

    # 创建精灵组
    all_sprites = pygame.sprite.Group()
    platforms = pygame.sprite.Group()
    coins = pygame.sprite.Group()

    # 创建芦泽
    芦泽 = Player()
    all_sprites.add(芦泽)

    # 创建平台
    ground = Platform(0, HEIGHT-40, WIDTH, 40)
    platform1 = Platform(200, HEIGHT-150, 200, 20)
    platform2 = Platform(500, HEIGHT-250, 200, 20)
    platforms.add(ground, platform1, platform2)
    all_sprites.add(platforms)

    # 创建金币
    for _ in range(5):
        coin = Coin(300 + _*70, HEIGHT-180)
        coins.add(coin)
        all_sprites.add(coin)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 更新
        芦泽.update(platforms)

        # 金币收集检测
        collected = pygame.sprite.spritecollide(芦泽, coins, True)
        芦泽.score += len(collected) * 100

        # 界面绘制
        screen.fill((146, 244, 255))  # 天空蓝
        
        # 显示游戏信息
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"得分: {芦泽.score}", True, WHITE)
        lives_text = font.render(f"生命: {芦泽.lives}", True, WHITE)
        name_text = font.render("芦泽", True, WHITE)
        
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 50))
        screen.blit(name_text, (芦泽.rect.x, 芦泽.rect.y - 30))
        
        all_sprites.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
