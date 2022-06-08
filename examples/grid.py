import pygame

TITLE = "Grid"
TILES_HORIZONTAL = 10
TILES_VERTICAL = 10
TILE_SIZE = 128
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 1280


class Player:
    def __init__(self, surface):
        self.surface = surface
        self.pos = (64, 64)

    def draw(self):
        pygame.draw.circle(self.surface, (255, 255, 255), self.pos, 64)

    def move(self, target):
        x = (128 * (target[0] // 128)) + 64
        y = (128 *(target[1] // 128)) + 64

        self.pos = (x, y)


class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(TITLE)
        self.surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.loop = True
        self.player = Player(self.surface)

    def main(self):
        while self.loop:
            self.events()
            self.draw()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.loop = False
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                self.player.move(pos)

    def draw(self):
        self.surface.fill((0, 0, 0))
        for row in range(TILES_HORIZONTAL):
            for col in range(row % 2, TILES_HORIZONTAL, 2):
                pygame.draw.rect(self.surface, (40, 40, 40), (row * TILE_SIZE, col * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        self.player.draw()
        pygame.display.update()


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
