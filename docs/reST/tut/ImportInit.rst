import math
import random
import sys
from dataclasses import dataclass

import pygame

# ============================================================
# CONFIGURACIÓN
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 60

PLAYER_MAX_HP = 200
PLAYER_SPEED = 280.0
PLAYER_RADIUS = 18
PLAYER_FIRE_RATE = 0.09
PLAYER_BULLET_SPEED = 800
PLAYER_BULLET_DAMAGE = 28

ENEMY_RADIUS = 16
ENEMY_SPEED = 95.0
ENEMY_MAX_HP = 100
ENEMY_FIRE_RATE = 0.85
ENEMY_BULLET_SPEED = 420
ENEMY_BULLET_DAMAGE = 10

LOOT_HEAL = 40
LOOT_LIFETIME = 20.0

GLOO_DURATION = 5.0
GLOO_COOLDOWN = 8.0
GLOO_DISTANCE = 65
GLOO_WIDTH = 100
GLOO_HEIGHT = 24

ZONE_START_RADIUS = 470
ZONE_MIN_RADIUS = 110
ZONE_SHRINK_DURATION = 150.0
ZONE_DAMAGE_PER_SECOND = 12.0

ENEMY_SPAWN_MIN_TIME = 0.8
ENEMY_SPAWN_MAX_TIME = 2.0

# Colores
BG = (28, 33, 38)
GROUND = (45, 65, 48)
GRID = (59, 83, 61)

WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
RED = (220, 60, 60)
DARK_RED = (120, 30, 30)
GREEN = (70, 220, 100)
DARK_GREEN = (35, 110, 50)
BLUE = (65, 150, 255)
CYAN = (50, 210, 240)
YELLOW = (250, 210, 60)
ORANGE = (245, 130, 40)
PURPLE = (190, 90, 240)
GRAY = (120, 125, 130)
DARK_GRAY = (55, 58, 62)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def circle_collision(pos1, radius1, pos2, radius2):
    return distance(pos1, pos2) <= radius1 + radius2


def normalize(vector):
    length = vector.length()
    if length == 0:
        return pygame.Vector2(0, 0)
    return vector / length


def draw_text(surface, font, text, position, color=WHITE):
    surface.blit(font.render(text, True, color), position)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Bullet:
    position: pygame.Vector2
    velocity: pygame.Vector2
    damage: int
    owner: str
    radius: int = 5
    alive: bool = True

    def update(self, dt):
        self.position += self.velocity * dt

        if (
            self.position.x < -30
            or self.position.x > WIDTH + 30
            or self.position.y < -30
            or self.position.y > HEIGHT + 30
        ):
            self.alive = False


@dataclass
class LootBox:
    position: pygame.Vector2
    lifetime: float = LOOT_LIFETIME
    alive: bool = True

    def update(self, dt):
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False


@dataclass
class GlooWall:
    position: pygame.Vector2
    direction: pygame.Vector2
    lifetime: float = GLOO_DURATION

    def update(self, dt):
        self.lifetime -= dt

    @property
    def alive(self):
        return self.lifetime > 0

    def get_rect(self):
        angle = math.degrees(math.atan2(self.direction.y, self.direction.x))

        rect = pygame.Rect(
            0,
            0,
            GLOO_WIDTH,
            GLOO_HEIGHT,
        )
        rect.center = (int(self.position.x), int(self.position.y))

        # El rectángulo rotado se crea usando una superficie temporal.
        temp = pygame.Surface((GLOO_WIDTH + 20, GLOO_HEIGHT + 20), pygame.SRCALPHA)
        temp_rect = temp.get_rect(center=rect.center)
        rotated = pygame.transform.rotate(temp, -angle)

        return rotated.get_rect(center=temp_rect.center)


# ============================================================
# JUGADOR
# ============================================================

class Player:
    def __init__(self):
        self.position = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        self.hp = PLAYER_MAX_HP
        self.angle = 0.0

        self.fire_timer = 0.0

        self.gloo_cooldown = 0.0
        self.gloo_walls = []

        self.alive = True
        self.kills = 0

    def handle_movement(self, dt):
        keys = pygame.key.get_pressed()

        direction = pygame.Vector2(0, 0)

        if keys[pygame.K_w]:
            direction.y -= 1
        if keys[pygame.K_s]:
            direction.y += 1
        if keys[pygame.K_a]:
            direction.x -= 1
        if keys[pygame.K_d]:
            direction.x += 1

        if direction.length_squared() > 0:
            direction = normalize(direction)
            self.position += direction * PLAYER_SPEED * dt

        self.position.x = clamp(
            self.position.x,
            PLAYER_RADIUS,
            WIDTH - PLAYER_RADIUS,
        )

        self.position.y = clamp(
            self.position.y,
            PLAYER_RADIUS,
            HEIGHT - PLAYER_RADIUS,
        )

    def update(self, dt, mouse_position):
        self.handle_movement(dt)

        mouse = pygame.Vector2(mouse_position)
        aim = mouse - self.position

        if aim.length_squared() > 0:
            self.angle = math.atan2(aim.y, aim.x)

        if self.fire_timer > 0:
            self.fire_timer -= dt

        if self.gloo_cooldown > 0:
            self.gloo_cooldown -= dt

        for wall in self.gloo_walls:
            wall.update(dt)

        self.gloo_walls = [wall for wall in self.gloo_walls if wall.alive]

    def shoot(self):
        if self.fire_timer > 0 or not self.alive:
            return None

        direction = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle),
        )

        self.fire_timer = PLAYER_FIRE_RATE

        spawn_position = self.position + direction * 24

        return Bullet(
            position=spawn_position,
            velocity=direction * PLAYER_BULLET_SPEED,
            damage=PLAYER_BULLET_DAMAGE,
            owner="player",
            radius=5,
        )

    def deploy_gloo(self):
        if self.gloo_cooldown > 0 or not self.alive:
            return False

        direction = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle),
        )

        wall_position = self.position + direction * GLOO_DISTANCE

        # Evita crear el muro fuera del mapa.
        wall_position.x = clamp(wall_position.x, 60, WIDTH - 60)
        wall_position.y = clamp(wall_position.y, 60, HEIGHT - 60)

        wall = GlooWall(
            position=wall_position,
            direction=direction,
        )

        self.gloo_walls.append(wall)
        self.gloo_cooldown = GLOO_COOLDOWN
        return True

    def take_damage(self, amount):
        self.hp -= amount
        self.hp = max(0, self.hp)

        if self.hp <= 0:
            self.alive = False

    def heal(self, amount):
        self.hp = min(PLAYER_MAX_HP, self.hp + amount)


# ============================================================
# ENEMIGO
# ============================================================

class Enemy:
    def __init__(self, position):
        self.position = pygame.Vector2(position)
        self.hp = ENEMY_MAX_HP

        self.fire_timer = random.uniform(0.2, ENEMY_FIRE_RATE)
        self.alive = True

        # Pequeña variación para que no todos sean idénticos.
        self.speed = random.uniform(
            ENEMY_SPEED * 0.8,
            ENEMY_SPEED * 1.15,
        )

        self.strafe_direction = random.choice([-1, 1])
        self.ai_timer = random.uniform(0.5, 1.5)

    def update(self, dt, player):
        if not self.alive:
            return None

        to_player = player.position - self.position
        dist = to_player.length()

        if dist > 0:
            direction = normalize(to_player)

            self.ai_timer -= dt

            if self.ai_timer <= 0:
                self.strafe_direction *= -1
                self.ai_timer = random.uniform(0.8, 2.0)

            # Movimiento de persecución + pequeño desplazamiento lateral.
            perpendicular = pygame.Vector2(
                -direction.y,
                direction.x,
            )

            movement = direction

            if dist < 260:
                movement += perpendicular * 0.35 * self.strafe_direction

            movement = normalize(movement)

            # Mantener distancia mínima.
            if dist > 150:
                self.position += movement * self.speed * dt
            elif dist < 95:
                self.position -= direction * self.speed * 0.65 * dt
            else:
                self.position += perpendicular * self.strafe_direction * self.speed * 0.25 * dt

        self.position.x = clamp(
            self.position.x,
            ENEMY_RADIUS,
            WIDTH - ENEMY_RADIUS,
        )

        self.position.y = clamp(
            self.position.y,
            ENEMY_RADIUS,
            HEIGHT - ENEMY_RADIUS,
        )

        if self.fire_timer > 0:
            self.fire_timer -= dt

        # Disparo automático cuando está en rango.
        if dist < 550 and self.fire_timer <= 0:
            self.fire_timer = ENEMY_FIRE_RATE + random.uniform(
                -0.15,
                0.15,
            )

            direction = normalize(player.position - self.position)

            return Bullet(
                position=self.position + direction * 20,
                velocity=direction * ENEMY_BULLET_SPEED,
                damage=ENEMY_BULLET_DAMAGE,
                owner="enemy",
                radius=5,
            )

        return None

    def take_damage(self, amount):
        self.hp -= amount

        if self.hp <= 0:
            self.hp = 0
            self.alive = False


# ============================================================
# JUEGO
# ============================================================

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Top-Down Battle Royale Prototype")

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "arial",
            22,
            bold=True,
        )

        self.small_font = pygame.font.SysFont(
            "arial",
            17,
        )

        self.big_font = pygame.font.SysFont(
            "arial",
            52,
            bold=True,
        )

        self.reset()

    def reset(self):
        self.player = Player()

        self.enemies = []
        self.bullets = []
        self.loot = []

        self.game_time = 0.0

        self.spawn_timer = 1.5

        self.zone_center = pygame.Vector2(
            WIDTH / 2,
            HEIGHT / 2,
        )

        self.zone_radius = ZONE_START_RADIUS
        self.zone_time_left = ZONE_SHRINK_DURATION

        self.game_over = False
        self.victory = False

    # --------------------------------------------------------
    # SPAWN DE ENEMIGOS
    # --------------------------------------------------------

    def spawn_enemy(self):
        side = random.randint(0, 3)
        margin = 20

        if side == 0:
            position = (
                random.randint(0, WIDTH),
                -margin,
            )
        elif side == 1:
            position = (
                WIDTH + margin,
                random.randint(0, HEIGHT),
            )
        elif side == 2:
            position = (
                random.randint(0, WIDTH),
                HEIGHT + margin,
            )
        else:
            position = (
                -margin,
                random.randint(0, HEIGHT),
            )

        self.enemies.append(
            Enemy(position)
        )

    # --------------------------------------------------------
    # ZONA
    # --------------------------------------------------------

    def update_zone(self, dt):
        self.zone_time_left -= dt

        progress = 1.0 - (
            max(self.zone_time_left, 0) / ZONE_SHRINK_DURATION
        )

        self.zone_radius = (
            ZONE_START_RADIUS
            - (
                ZONE_START_RADIUS - ZONE_MIN_RADIUS
            )
            * progress
        )

        if (
            distance(
                self.player.position,
                self.zone_center,
            )
            > self.zone_radius
        ):
            self.player.take_damage(
                ZONE_DAMAGE_PER_SECOND * dt
            )

    # --------------------------------------------------------
    # COLISIÓN DE BALAS CON MURO
    # --------------------------------------------------------

    def bullet_hits_wall(self, bullet):
        for wall in self.player.gloo_walls:
            rect = wall.get_rect()

            if rect.collidepoint(
                int(bullet.position.x),
                int(bullet.position.y),
            ):
                return True

        return False

    # --------------------------------------------------------
    # ACTUALIZAR BALAS
    # --------------------------------------------------------

    def update_bullets(self, dt):
        for bullet in self.bullets:
            if not bullet.alive:
                continue

            bullet.update(dt)

            if not bullet.alive:
                continue

            # Las paredes bloquean las balas.
            if self.bullet_hits_wall(bullet):
                bullet.alive = False
                continue

            if bullet.owner == "player":
                # Bala del jugador contra enemigos.
                for enemy in self.enemies:
                    if not enemy.alive:
                        continue

                    if circle_collision(
                        bullet.position,
                        bullet.radius,
                        enemy.position,
                        ENEMY_RADIUS,
                    ):
                        enemy.take_damage(
                            bullet.damage
                        )

                        bullet.alive = False

                        if not enemy.alive:
                            self.player.kills += 1

                            self.loot.append(
                                LootBox(
                                    position=pygame.Vector2(
                                        enemy.position
                                    )
                                )
                            )

                        break

            else:
                # Bala enemiga contra jugador.
                if circle_collision(
                    bullet.position,
                    bullet.radius,
                    self.player.position,
                    PLAYER_RADIUS,
                ):
                    self.player.take_damage(
                        bullet.damage
                    )

                    bullet.alive = False

        self.bullets = [
            bullet
            for bullet in self.bullets
            if bullet.alive
        ]

    # --------------------------------------------------------
    # ACTUALIZAR LOOT
    # --------------------------------------------------------

    def update_loot(self, dt):
        for item in self.loot:
            if not item.alive:
                continue

            item.update(dt)

            if not item.alive:
                continue

            if circle_collision(
                item.position,
                18,
                self.player.position,
                PLAYER_RADIUS,
            ):
                self.player.heal(
                    LOOT_HEAL
                )

                item.alive = False

        self.loot = [
            item
            for item in self.loot
            if item.alive
        ]

    # --------------------------------------------------------
    # ACTUALIZAR ENEMIGOS
    # --------------------------------------------------------

    def update_enemies(self, dt):
        for enemy in self.enemies:
            if not enemy.alive:
                continue

            bullet = enemy.update(
                dt,
                self.player,
            )

            if bullet is not None:
                self.bullets.append(
                    bullet
                )

        self.enemies = [
            enemy
            for enemy in self.enemies
            if enemy.alive
        ]

    # --------------------------------------------------------
    # ACTUALIZACIÓN GENERAL
    # --------------------------------------------------------

    def update(self, dt):
        if self.game_over or self.victory:
            return

        self.game_time += dt

        mouse_pos = pygame.mouse.get_pos()

        self.player.update(
            dt,
            mouse_pos,
        )

        # Spawn progresivo.
        self.spawn_timer -= dt

        if self.spawn_timer <= 0:
            self.spawn_enemy()

            # Aumenta gradualmente la frecuencia.
            difficulty = min(
                self.game_time / 120.0,
                1.0,
            )

            minimum = (
                ENEMY_SPAWN_MIN_TIME
                - difficulty * 0.3
            )

            maximum = (
                ENEMY_SPAWN_MAX_TIME
                - difficulty * 0.5
            )

            self.spawn_timer = random.uniform(
                max(0.3, minimum),
                max(0.5, maximum),
            )

        self.update_enemies(dt)
        self.update_bullets(dt)
        self.update_loot(dt)
        self.update_zone(dt)

        # Limpiar enemigos muertos.
        self.enemies = [
            enemy
            for enemy in self.enemies
            if enemy.alive
        ]

        if not self.player.alive:
            self.game_over = True

        # Victoria solamente si sobrevive y no quedan enemigos
        # después de haber transcurrido un tiempo inicial.
        if self.game_time > 8 and len(self.enemies) == 0:
            # En lugar de victoria inmediata, seguimos generando
            # enemigos para mantener la partida viva.
            pass

    # --------------------------------------------------------
    # DIBUJAR FONDO
    # --------------------------------------------------------

    def draw_background(self):
        self.screen.fill(GROUND)

        grid_size = 48

        for x in range(0, WIDTH, grid_size):
            pygame.draw.line(
                self.screen,
                GRID,
                (x, 0),
                (x, HEIGHT),
                1,
            )

        for y in range(0, HEIGHT, grid_size):
            pygame.draw.line(
                self.screen,
                GRID,
                (0, y),
                (WIDTH, y),
                1,
            )

        # Decoración sencilla del terreno.
        random.seed(12345)

        for _ in range(65):
            x = random.randint(20, WIDTH - 20)
            y = random.randint(20, HEIGHT - 20)

            pygame.draw.circle(
                self.screen,
                (40, 78, 43),
                (x, y),
                random.randint(2, 5),
            )

        random.seed()

    # --------------------------------------------------------
    # DIBUJAR ZONA
    # --------------------------------------------------------

    def draw_zone(self):
        # Oscurecer el exterior de la zona.
        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (100, 30, 120, 100)
        )

        pygame.draw.circle(
            overlay,
            (0, 0, 0, 0),
            (
                int(self.zone_center.x),
                int(self.zone_center.y),
            ),
            int(self.zone_radius),
        )

        self.screen.blit(
            overlay,
            (0, 0),
        )

        # Borde de la zona segura.
        pygame.draw.circle(
            self.screen,
            CYAN,
            (
                int(self.zone_center.x),
                int(self.zone_center.y),
            ),
            int(self.zone_radius),
            5,
        )

    # --------------------------------------------------------
    # DIBUJAR LOOT
    # --------------------------------------------------------

    def draw_loot(self):
        for item in self.loot:
            x = int(item.position.x)
            y = int(item.position.y)

            # Sombra.
            pygame.draw.ellipse(
                self.screen,
                (20, 25, 20),
                pygame.Rect(
                    x - 13,
                    y + 9,
                    26,
                    8,
                ),
            )

            # Caja.
            pygame.draw.rect(
                self.screen,
                ORANGE,
                pygame.Rect(
                    x - 13,
                    y - 13,
                    26,
                    26,
                ),
                border_radius=5,
            )

            pygame.draw.rect(
                self.screen,
                YELLOW,
                pygame.Rect(
                    x - 13,
                    y - 13,
                    26,
                    26,
                ),
                2,
                border_radius=5,
            )

            # Cruz médica.
            pygame.draw.rect(
                self.screen,
                WHITE,
                pygame.Rect(
                    x - 3,
                    y - 9,
                    6,
                    18,
                ),
            )

            pygame.draw.rect(
                self.screen,
                WHITE,
                pygame.Rect(
                    x - 9,
                    y - 3,
                    18,
                    6,
                ),
            )

    # --------------------------------------------------------
    # DIBUJAR ENEMIGOS
    # --------------------------------------------------------

    def draw_enemies(self):
        for enemy in self.enemies:
            x = int(enemy.position.x)
            y = int(enemy.position.y)

            # Sombra.
            pygame.draw.ellipse(
                self.screen,
                (20, 20, 20),
                pygame.Rect(
                    x - 15,
                    y + 10,
                    30,
                    9,
                ),
            )

            # Cuerpo.
            pygame.draw.circle(
                self.screen,
                RED,
                (x, y),
                ENEMY_RADIUS,
            )

            pygame.draw.circle(
                self.screen,
                WHITE,
                (x, y),
                5,
            )

            pygame.draw.circle(
                self.screen,
                BLACK,
                (x, y),
                2,
            )

            # Barra HP.
            bar_width = 34
            bar_height = 5

            hp_ratio = enemy.hp / ENEMY_MAX_HP

            pygame.draw.rect(
                self.screen,
                DARK_RED,
                pygame.Rect(
                    x - bar_width // 2,
                    y - 27,
                    bar_width,
                    bar_height,
                ),
            )

            pygame.draw.rect(
                self.screen,
                GREEN,
                pygame.Rect(
                    x - bar_width // 2,
                    y - 27,
                    int(bar_width * hp_ratio),
                    bar_height,
                ),
            )

    # --------------------------------------------------------
    # DIBUJAR PARED DE GLOO
    # --------------------------------------------------------

    def draw_gloo_walls(self):
        for wall in self.player.gloo_walls:
            angle = math.degrees(
                math.atan2(
                    wall.direction.y,
                    wall.direction.x,
                )
            )

            width = GLOO_WIDTH
            height = GLOO_HEIGHT

            surface = pygame.Surface(
                (width + 30, height + 30),
                pygame.SRCALPHA,
            )

            pygame.draw.rect(
                surface,
                (70, 220, 255, 210),
                pygame.Rect(
                    15,
                    15,
                    width,
                    height,
                ),
                border_radius=10,
            )

            pygame.draw.rect(
                surface,
                (220, 250, 255, 240),
                pygame.Rect(
                    15,
                    15,
                    width,
                    height,
                ),
                3,
                border_radius=10,
            )

            rotated = pygame.transform.rotate(
                surface,
                -angle,
            )

            rect = rotated.get_rect(
                center=(
                    int(wall.position.x),
                    int(wall.position.y),
                )
            )

            self.screen.blit(
                rotated,
                rect,
            )

    # --------------------------------------------------------
    # DIBUJAR JUGADOR
    # --------------------------------------------------------

    def draw_player(self):
        x = int(self.player.position.x)
        y = int(self.player.position.y)

        # Sombra.
        pygame.draw.ellipse(
            self.screen,
            (20, 20, 20),
            pygame.Rect(
                x - 18,
                y + 12,
                36,
                10,
            ),
        )

        # Cuerpo.
        pygame.draw.circle(
            self.screen,
            BLUE,
            (x, y),
            PLAYER_RADIUS,
        )

        pygame.draw.circle(
            self.screen,
            (180, 220, 255),
            (x, y),
            PLAYER_RADIUS,
            3,
        )

        # Dirección del arma.
        direction = pygame.Vector2(
            math.cos(self.player.angle),
            math.sin(self.player.angle),
        )

        weapon_end = (
            self.player.position
            + direction * 30
        )

        pygame.draw.line(
            self.screen,
            BLACK,
            (x, y),
            (
                int(weapon_end.x),
                int(weapon_end.y),
            ),
            8,
        )

        pygame.draw.line(
            self.screen,
            WHITE,
            (x, y),
            (
                int(weapon_end.x),
                int(weapon_end.y),
            ),
            3,
        )

    # --------------------------------------------------------
    # DIBUJAR BALAS
    # --------------------------------------------------------

    def draw_bullets(self):
        for bullet in self.bullets:
            color = (
                YELLOW
                if bullet.owner == "player"
                else RED
            )

            pygame.draw.circle(
                self.screen,
                color,
                (
                    int(bullet.position.x),
                    int(bullet.position.y),
                ),
                bullet.radius,
            )

    # --------------------------------------------------------
    # HUD
    # --------------------------------------------------------

    def draw_hud(self):
        panel = pygame.Surface(
            (360, 130),
            pygame.SRCALPHA,
        )

        panel.fill(
            (10, 10, 15, 190)
        )

        self.screen.blit(
            panel,
            (15, 15),
        )

        # Texto HP.
        draw_text(
            self.screen,
            self.font,
            f"HP: {int(self.player.hp)} / {PLAYER_MAX_HP}",
            (30, 25),
        )

        # Barra de vida.
        hp_ratio = self.player.hp / PLAYER_MAX_HP

        pygame.draw.rect(
            self.screen,
            DARK_RED,
            pygame.Rect(
                30,
                55,
                320,
                22,
            ),
            border_radius=6,
        )

        pygame.draw.rect(
            self.screen,
            GREEN,
            pygame.Rect(
                30,
                55,
                int(320 * hp_ratio),
                22,
            ),
            border_radius=6,
        )

        pygame.draw.rect(
            self.screen,
            WHITE,
            pygame.Rect(
                30,
                55,
                320,
                22,
            ),
            2,
            border_radius=6,
        )

        # Enemigos.
        draw_text(
            self.screen,
            self.font,
            f"Enemigos: {len(self.enemies)}",
            (30, 88),
        )

        # Tiempo de zona.
        zone_seconds = max(
            0,
            int(self.zone_time_left),
        )

        draw_text(
            self.screen,
            self.font,
            f"Zona: {zone_seconds}s",
            (190, 88),
        )

        # Gloo.
        if self.player.gloo_cooldown <= 0:
            gloo_text = "Gloo [E]: LISTO"
            gloo_color = CYAN
        else:
            gloo_text = (
                f"Gloo [E]: "
                f"{self.player.gloo_cooldown:.1f}s"
            )
            gloo_color = WHITE

        draw_text(
            self.screen,
            self.small_font,
            gloo_text,
            (30, 116),
            gloo_color,
        )

        # Kills.
        draw_text(
            self.screen,
            self.small_font,
            f"Bajas: {self.player.kills}",
            (220, 116),
        )

        # Controles en la parte inferior.
        controls = (
            "WASD: mover   "
            "Mouse: apuntar/disparar   "
            "E: pared Gloo"
        )

        text_surface = self.small_font.render(
            controls,
            True,
            WHITE,
        )

        rect = text_surface.get_rect(
            center=(WIDTH // 2, HEIGHT - 18)
        )

        self.screen.blit(
            text_surface,
            rect,
        )

    # --------------------------------------------------------
    # PANTALLA DE GAME OVER
    # --------------------------------------------------------

    def draw_game_over(self):
        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (0, 0, 0, 175)
        )

        self.screen.blit(
            overlay,
            (0, 0),
        )

        title = "HAS SIDO ELIMINADO"

        title_surface = self.big_font.render(
            title,
            True,
            RED,
        )

        title_rect = title_surface.get_rect(
            center=(
                WIDTH // 2,
                HEIGHT // 2 - 45,
            )
        )

        self.screen.blit(
            title_surface,
            title_rect,
        )

        draw_text(
            self.screen,
            self.font,
            f"Bajas: {self.player.kills}",
            (
                WIDTH // 2 - 50,
                HEIGHT // 2 + 25,
            ),
        )

        draw_text(
            self.screen,
            self.font,
            "Pulsa R para reiniciar",
            (
                WIDTH // 2 - 115,
                HEIGHT // 2 + 65,
            ),
        )

    # --------------------------------------------------------
    # DIBUJAR TODO
    # --------------------------------------------------------

    def draw(self):
        self.draw_background()
        self.draw_zone()
        self.draw_loot()
        self.draw_gloo_walls()
        self.draw_enemies()
        self.draw_bullets()
        self.draw_player()
        self.draw_hud()

        if self.game_over:
            self.draw_game_over()

    # --------------------------------------------------------
    # EVENTOS
    # --------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_e:
                    self.player.deploy_gloo()

                if (
                    event.key == pygame.K_r
                    and self.game_over
                ):
                    self.reset()

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                bullet = self.player.shoot()

                if bullet:
                    self.bullets.append(
                        bullet
                    )

    # --------------------------------------------------------
    # DISPARO CONTINUO
    # --------------------------------------------------------

    def handle_continuous_fire(self):
        if self.game_over:
            return

        buttons = pygame.mouse.get_pressed()

        if buttons[0]:
            bullet = self.player.shoot()

            if bullet:
                self.bullets.append(
                    bullet
                )

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            # Evita saltos gigantes si la ventana se congela.
            dt = min(dt, 0.05)

            self.handle_events()
            self.handle_continuous_fire()

            self.update(dt)
            self.draw()

            pygame.display.flip()


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
