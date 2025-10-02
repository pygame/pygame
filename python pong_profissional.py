import pygame
import sys

# Inicialização do Pygame
pygame.init()

# === CONFIGURAÇÕES ===
LARGURA_TELA = 900
ALTURA_TELA = 600
COR_FUNDO = (30, 30, 30)
COR_RAQUETE = (240, 240, 240)
COR_BOLA = (200, 50, 50)
COR_PONTUACAO = (255, 255, 255)

# Velocidades
VELOCIDADE_RAQUETE = 7
VELOCIDADE_BOLA_INICIAL = 6

# Fonte para pontuação
FONTE_PONTUACAO = pygame.font.SysFont("Arial", 48)

# Criar a janela
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Pong Profissional - ChatGPT")

# Relógio para FPS
clock = pygame.time.Clock()
FPS = 60

# === CLASSES ===

class Raquete:
    def __init__(self, x_pos):
        self.largura = 15
        self.altura = 100
        self.x = x_pos
        self.y = ALTURA_TELA // 2 - self.altura // 2
        self.velocidade = 0

    def desenhar(self):
        pygame.draw.rect(tela, COR_RAQUETE, (self.x, self.y, self.largura, self.altura))

    def mover(self):
        self.y += self.velocidade
        # Limitar para não sair da tela
        if self.y < 0:
            self.y = 0
        elif self.y + self.altura > ALTURA_TELA:
            self.y = ALTURA_TELA - self.altura

class Bola:
    def __init__(self):
        self.tamanho = 20
        self.resetar()

    def resetar(self):
        self.x = LARGURA_TELA // 2 - self.tamanho // 2
        self.y = ALTURA_TELA // 2 - self.tamanho // 2
        self.vel_x = VELOCIDADE_BOLA_INICIAL * (-1 if pygame.time.get_ticks() % 2 == 0 else 1)
        self.vel_y = VELOCIDADE_BOLA_INICIAL * (-1 if pygame.time.get_ticks() % 3 == 0 else 1)

    def desenhar(self):
        pygame.draw.ellipse(tela, COR_BOLA, (self.x, self.y, self.tamanho, self.tamanho))

    def mover(self):
        self.x += self.vel_x
        self.y += self.vel_y

        # Rebate no topo e no chão
        if self.y <= 0 or self.y + self.tamanho >= ALTURA_TELA:
            self.vel_y *= -1

# === Funções auxiliares ===

def desenhar_pontuacao(pontos_esquerda, pontos_direita):
    texto_esquerda = FONTE_PONTUACAO.render(str(pontos_esquerda), True, COR_PONTUACAO)
    texto_direita = FONTE_PONTUACAO.render(str(pontos_direita), True, COR_PONTUACAO)
    tela.blit(texto_esquerda, (LARGURA_TELA // 4 - texto_esquerda.get_width() // 2, 20))
    tela.blit(texto_direita, (LARGURA_TELA * 3 // 4 - texto_direita.get_width() // 2, 20))

def desenhar_centro():
    # Linha pontilhada no meio
    segmento_altura = 15
    segmento_espaco = 10
    y = 0
    while y < ALTURA_TELA:
        pygame.draw.rect(tela, COR_PONTUACAO, (LARGURA_TELA // 2 - 5, y, 10, segmento_altura))
        y += segmento_altura + segmento_espaco

# === Inicializa objetos ===
raquete_esquerda = Raquete(30)
raquete_direita = Raquete(LARGURA_TELA - 30 - 15)
bola = Bola()

pontos_esquerda = 0
pontos_direita = 0

# === Loop principal ===
rodando = True
while rodando:
    clock.tick(FPS)
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            rodando = False
        # Detecta teclas pressionadas
        elif evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_w:
                raquete_esquerda.velocidade = -VELOCIDADE_RAQUETE
            elif evento.key == pygame.K_s:
                raquete_esquerda.velocidade = VELOCIDADE_RAQUETE
            elif evento.key == pygame.K_UP:
                raquete_direita.velocidade = -VELOCIDADE_RAQUETE
            elif evento.key == pygame.K_DOWN:
                raquete_direita.velocidade = VELOCIDADE_RAQUETE
        elif evento.type == pygame.KEYUP:
            if evento.key in [pygame.K_w, pygame.K_s]:
                raquete_esquerda.velocidade = 0
            elif evento.key in [pygame.K_UP, pygame.K_DOWN]:
                raquete_direita.velocidade = 0

    # Movimento
    raquete_esquerda.mover()
    raquete_direita.mover()
    bola.mover()

    # Colisão da bola com as raquetes
    ret_bola = pygame.Rect(bola.x, bola.y, bola.tamanho, bola.tamanho)
    ret_raquete_esq = pygame.Rect(raquete_esquerda.x, raquete_esquerda.y, raquete_esquerda.largura, raquete_esquerda.altura)
    ret_raquete_dir = pygame.Rect(raquete_direita.x, raquete_direita.y, raquete_direita.largura, raquete_direita.altura)

    if ret_bola.colliderect(ret_raquete_esq):
        bola.vel_x *= -1.1  # aumenta a velocidade levemente
        bola.x = raquete_esquerda.x + raquete_esquerda.largura

    if ret_bola.colliderect(ret_raquete_dir):
        bola.vel_x *= -1.1
        bola.x = raquete_direita.x - bola.tamanho

    # Pontuação e reset da bola
    if bola.x < 0:
        pontos_direita += 1
        bola.resetar()
    elif bola.x > LARGURA_TELA:
        pontos_esquerda += 1
        bola.resetar()

    # Desenho da tela
    tela.fill(COR_FUNDO)
    desenhar_centro()
    raquete_esquerda.desenhar()
    raquete_direita.desenhar()
    bola.desenhar()
    desenhar_pontuacao(pontos_esquerda, pontos_direita)

    pygame.display.flip()

pygame.quit()
sys.exit()
