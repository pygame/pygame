import pygame as pg

TITLE = "Grid"
TILES_HORIZONTAL = 10
TILES_VERTICAL = 10
TILE_SIZE = 100
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000


class Player:
    def __init__(self, surface):
        self.surface = surface
        self.pos = (50, 50)

    def draw(self):
        pg.draw.circle(self.surface, (255, 255, 255), self.pos, 50)

    def move(self, target):
        x = (100 * (target[0] // 100)) + 50
        y = (100 * (target[1] // 100)) + 50

        self.pos = (x, y)


class Game:
    def __init__(self):
        pg.init()
        self.clock = pg.time.Clock()
        pg.display.set_caption(TITLE)
        self.surface = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.loop = True
        self.player = Player(self.surface)

    def main(self):
        while self.loop:
            self.grid_loop()

    def grid_loop(self):
        self.surface.fill((0, 0, 0))
        for row in range(TILES_HORIZONTAL):
            for col in range(row % 2, TILES_HORIZONTAL, 2):
                pg.draw.rect(self.surface, (40, 40, 40), (row * TILE_SIZE, col * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        self.player.draw()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.loop = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.loop = False
            elif event.type == pg.MOUSEBUTTONUP:
                pos = pg.mouse.get_pos()
                self.player.move(pos)
        pg.display.update()


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
    
