# python -m pygame.docs

try:
    import util
except ModuleNotFoundError:
    import docs.util as util
util.open_docs()
