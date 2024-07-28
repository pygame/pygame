import pygame
import random
import time

# 初始化 Pygame
pygame.init()

# 設置遊戲窗口
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("翻牌遊戲 - 縱火犯ARSONATE")

# 定義顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 卡片設置
CARD_WIDTH, CARD_HEIGHT = 100, 150
MARGIN = 20

# 加載圖片
fire_img = pygame.image.load('fire.png')
lighter_img = pygame.image.load('lighter.png')
gasoline_img = pygame.image.load('gasoline.png')
cards = [fire_img, lighter_img, gasoline_img] * 2  # 每個圖案兩張

# 打亂卡片
random.shuffle(cards)

# 創建卡片矩形列表
card_rects = []
for i in range(3):
    for j in range(2):
        x = MARGIN + j * (CARD_WIDTH + MARGIN)
        y = MARGIN + i * (CARD_HEIGHT + MARGIN)
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        card_rects.append(rect)

# 遊戲狀態
flipped = [False] * 6
first_card = None
second_card = None
match_count = 0

# 遊戲主循環
running = True
while running:
    win.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for i, rect in enumerate(card_rects):
                if rect.collidepoint(pos) and not flipped[i]:
                    if not first_card:
                        first_card = i
                        flipped[i] = True
                    elif not second_card:
                        second_card = i
                        flipped[i] = True

    if first_card is not None and second_card is not None:
        if cards[first_card] == cards[second_card]:
            match_count += 1
        else:
            time.sleep(1)
            flipped[first_card] = False
            flipped[second_card] = False
        first_card, second_card = None, None

    for i, rect in enumerate(card_rects):
        if flipped[i]:
            win.blit(cards[i], (rect.x, rect.y))
        else:
            pygame.draw.rect(win, BLACK, rect)

    pygame.display.flip()

    if match_count == 3:
        print("你贏了！")
        running = False

pygame.quit()

