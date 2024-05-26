import pygame
import random
import math

# Ekran genişliği ve yüksekliği
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Renkler
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Oyun sınıfı
class AngryBirdsGame:
    def _init_(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Angry Birds")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.target_pos = (random.randint(100, 700), random.randint(100, 500))
        self.score = 100

    # Yeni bir hedef oluştur
    def new_target(self):
        self.target_pos = (random.randint(100, 700), random.randint(100, 500))

    # Oyun döngüsü
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Fare tıklaması algılandığında atış yap
                    self.fire_bird(pygame.mouse.get_pos())

            self.screen.fill(WHITE)
            self.draw_target()

            # Puanı ekrana yazdır
            score_text = self.font.render("Score: " + str(self.score), True, BLACK)
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    # Kuş fırlatma fonksiyonu
    def fire_bird(self, mouse_pos):
        bird_pos = (50, SCREEN_HEIGHT - 50)
        distance = math.sqrt((mouse_pos[0] - self.target_pos[0])*2 + (mouse_pos[1] - self.target_pos[1])*2)

        if distance < 50:  # Hedefi vurduysa
            self.score += 50
            self.new_target()
        else:  # Hedefi vuramadıysa
            self.score -= 20

    # Hedefi çizme fonksiyonu
    def draw_target(self):
        pygame.draw.circle(self.screen, RED, self.target_pos, 30)
        pygame.draw.circle(self.screen, BLACK, self.target_pos, 30, 2)


if _name_ == "_main_":
    game = AngryBirdsGame()
    game.run()
