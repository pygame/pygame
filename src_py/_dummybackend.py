"""dummy Movie class if all else fails """
class Movie:
    def __init__(self, filename, surface=None):
        self.filename = filename
        self.surface = surface
        self.process = None
        self.loops = 0
        self.playing = False
        self.paused = False
        self._backend = "DUMMY"
        self.width = 0
        self.height = 0
        self.finished = 1

    def play(self, loops=0):
        self.playing = not self.playing

    def stop(self):
        self.playing = not self.playing
        self.paused = not self.paused

    def pause(self):
        self.paused = not self.paused

    def resize(self, w, h):
        self.width = w
        self.height = h

    def __repr__(self):
        return "(%s 0.0s)" % self.filename
