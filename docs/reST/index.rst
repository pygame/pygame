
import pygame, random, sys; pygame.init()
W, H = 400, 600; S = pygame.display.set_mode((W, H))
C = pygame.time.Clock(); G, J = 0.6, -15
px, py, vx, vy = 200, 500, 0, 0
P = [[200, 550]] + [[random.randint(0, 330), H - (i * 70) - 120] for i in range(8)]
S_PTS = 0; RUN = True

while RUN:
    C.tick(60); S.fill((0, 0, 0))
    for e in pygame.event.get():
        if e.type == pygame.QUIT: RUN = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_LEFT: vx = -6
            if e.key == pygame.K_RIGHT: vx = 6
        if e.type == pygame.KEYUP:
            if e.key in (pygame.K_LEFT, pygame.K_RIGHT): vx = 0
            
    vy += G; px += vx; py += vy
    px = -30 if px > W else (W if px < -30 else px)
    RJ = pygame.Rect(px, py, 30, 30)
    
    if vy > 0:
        for p in P:
            if RJ.colliderect(pygame.Rect(p[0], p[1], 70, 15)) and py + 30 - vy <= p[1] + 5:
                vy = J
                
    if py < 300:
        diff = 300 - py; py = 300; S_PTS += int(diff)
        for p in P: p[1] += diff
        
    for p in P[:]:
        if p[1] > H:
            P.remove(p)
            P.append([random.randint(0, 330), random.randint(-50, -10)])
            
    if py > H: RUN = False
    
    pygame.draw.rect(S, (50, 150, 255), RJ)
    for p in P: pygame.draw.rect(S, (100, 200, 50), (p[0], p[1], 70, 15))
    pygame.display.update()

pygame.quit(); sys.exit()
