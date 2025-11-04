import pygame
import random
import time

# 1. Configurações Iniciais do Pygame
pygame.init()

# 2. Definição de Cores (RGB)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (255, 0, 0)
VERDE = (0, 255, 0)

# 3. Configurações da Tela
largura = 600
altura = 400
tela = pygame.display.set_mode((largura, altura))
pygame.display.set_caption('Jogo da Cobrinha em Python')

# 4. Variáveis do Jogo
tamanho_bloco = 10
velocidade_cobra = 15
relogio = pygame.time.Clock()

# 5. Funções de Exibição
fonte_pontuacao = pygame.font.SysFont("bahnschrift", 25)
fonte_mensagem = pygame.font.SysFont("bahnschrift", 70)

def desenhar_cobra(tamanho_bloco, lista_cobra):
    """Desenha a cobra na tela."""
    for x in lista_cobra:
        pygame.draw.rect(tela, VERDE, [x[0], x[1], tamanho_bloco, tamanho_bloco])

def mensagem(msg, cor):
    """Exibe a mensagem de Game Over na tela."""
    texto = fonte_mensagem.render(msg, True, cor)
    tela.blit(texto, [largura / 6, altura / 3])

def mostrar_pontuacao(pontuacao):
    """Exibe a pontuação atual."""
    valor = fonte_pontuacao.render("Pontos: " + str(pontuacao), True, BRANCO)
    tela.blit(valor, [0, 0])

# 6. Loop Principal do Jogo
def game_loop():
    game_over = False
    game_fechar = False

    # Posição inicial da cabeça da cobra
    x1 = largura / 2
    y1 = altura / 2

    # Mudança de posição (inicialmente parada)
    x1_mudanca = 0
    y1_mudanca = 0

    # Lista de coordenadas do corpo e tamanho inicial
    lista_cobra = []
    comprimento_cobra = 1
    
    # Gerar a primeira comida em uma posição aleatória
    comida_x = round(random.randrange(0, largura - tamanho_bloco) / 10.0) * 10.0
    comida_y = round(random.randrange(0, altura - tamanho_bloco) / 10.0) * 10.0

    while not game_over:

        # Tela de Game Over
        while game_fechar == True:
            tela.fill(PRETO)
            mensagem("Você Perdeu!", VERMELHO)
            mostrar_pontuacao(comprimento_cobra - 1)
            pygame.display.update()

            for evento in pygame.event.get():
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_q: # Pressione 'Q' para sair
                        game_over = True
                        game_fechar = False
                    if evento.key == pygame.K_r: # Pressione 'R' para reiniciar
                        game_loop() # Reinicia o jogo
            
        # 7. Eventos (Controle do Teclado)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                game_over = True
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_LEFT and x1_mudanca == 0:
                    x1_mudanca = -tamanho_bloco
                    y1_mudanca = 0
                elif evento.key == pygame.K_RIGHT and x1_mudanca == 0:
                    x1_mudanca = tamanho_bloco
                    y1_mudanca = 0
                elif evento.key == pygame.K_UP and y1_mudanca == 0:
                    y1_mudanca = -tamanho_bloco
                    x1_mudanca = 0
                elif evento.key == pygame.K_DOWN and y1_mudanca == 0:
                    y1_mudanca = tamanho_bloco
                    x1_mudanca = 0

        # 8. Verificação de Colisão com as Bordas
        if x1 >= largura or x1 < 0 or y1 >= altura or y1 < 0:
            game_fechar = True

        # 9. Atualização da Posição
        x1 += x1_mudanca
        y1 += y1_mudanca
        tela.fill(PRETO) # Limpa a tela

        # Desenha a comida (quadrado vermelho)
        pygame.draw.rect(tela, VERMELHO, [comida_x, comida_y, tamanho_bloco, tamanho_bloco])
        
        # 10. Lógica da Cobra
        cabeca_cobra = []
        cabeca_cobra.append(x1)
        cabeca_cobra.append(y1)
        lista_cobra.append(cabeca_cobra)
        
        if len(lista_cobra) > comprimento_cobra:
            del lista_cobra[0]

        # 11. Verificação de Colisão com o Próprio Corpo
        for segmento in lista_cobra[:-1]:
            if segmento == cabeca_cobra:
                game_fechar = True

        desenhar_cobra(tamanho_bloco, lista_cobra)
        mostrar_pontuacao(comprimento_cobra - 1)

        pygame.display.update()

        # 12. Lógica de Comer a Comida
        if x1 == comida_x and y1 == comida_y:
            comida_x = round(random.randrange(0, largura - tamanho_bloco) / 10.0) * 10.0
            comida_y = round(random.randrange(0, altura - tamanho_bloco) / 10.0) * 10.0
            comprimento_cobra += 1

        # Controla a velocidade do jogo
        relogio.tick(velocidade_cobra)
        
    pygame.quit()
    quit()

# 7. Iniciar o Jogo
game_loop()

