# module trackmod.namereg

class NameRegistry(object):
    
    class AllRegistered(object):
        terminal = True
        def register(self, names):
            return
        def __contains__(self, name):
            return True
    all_registered = AllRegistered()

    class AllFound(object):
        def __init__(self, value):
            self.value = value
        def __getitem__(self, key):
            return self.value
    all_found = AllFound(all_registered)

    def __init__(self, names=None):
        self.names = {}
        if names is not None:
            self.add(names)
        self.terminal = False

    def add(self, names):
        if names is None:
            self.terminal = True
            return
        for name in names:
            parts = name.split('.', 1)
            first = parts[0]
            if first == '*':
                self.names = self.all_found
                return
            else:
                try:
                    sub_registry = self.names[first]
                except KeyError:
                    sub_registry = NameRegistry()
                    self.names[first] = sub_registry
                if len(parts) == 2:
                    sub_registry.add(parts[1:])
                else:
                    sub_registry.terminal = True

    def __contains__(self, name):
        parts = name.split('.', 1)
        try:
            sub_registry = self.names[parts[0]]
        except KeyError:
            return False
        # This uses a conditional or.
        if len(parts) == 1:
            return sub_registry.terminal
        return parts[1] in sub_registry

