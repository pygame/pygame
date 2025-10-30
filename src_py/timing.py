"""
-------------------------------------------------------------------------------------------------------
Time helper functions that I think would add to Pygame.
-------------------------------------------------------------------------------------------------------
"""
import pygame
from pygame.time import Clock
import time

class DeltaTime:
  def __init__(self):
    self._clock = Clock()
    self.dt = 0
  def tick(self, fps=0.0):
    self.dt = self._clock.tick(fps) / 1000.0
    return self.dt
class Stopwatch:
  def __init__(self):
    self._start = time.perf_counter()
    self._elapsed = 0.0
    self._running = False
    self.start()
  def start(self)
    if self.running:
      self._elapsed += time.perf_counter() - self._start
      self._running = False
  def reset(self):
    self._start = time.perf_counter()
  def elapsed(self):
    if self._running:
        return self._elapsed + (time.perf_counter() - self._start)
    return time.perf_counter() - self._start
    
class CooldownTimer:
  def __init__(self, cooldown: float):
    self.cooldown = cooldown
    self.timer = 0.0
  def update(self, dt: float):
    if self.timer > 0:
      self.timer -= dt
  def ready(self):
    return  self.timer <= 0
  def trigger(self):
    self.timer = self.cooldown
class Timer:
  def __init__(self, callback, delay):
    self.callback = callback
    self.delay = delay
    self.elapsed = 0.0
    self.finished = False
  def update(self, dt: float):
    if not self.finished:
      self.elapsed += dt
      if self.elapsed >= self.delay:
        self.callback()
        self.finished = True
    def reset(self)
      self.elapsed = 0.0
      self.finished = False
