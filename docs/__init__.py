# Make docs a package that brings up the main page in a web brower when
# executed.
#
# python -m pygame.docs

if __name__ == '__main__':
    import os
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    main = os.path.join(pkg_dir, '__main__.py')
    exec(open(main).read())


