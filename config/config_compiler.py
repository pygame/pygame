class CompilerDependency (object):
    """
    Generic compiler flag and dependency class.
    """
    def __init__ (self, name, cflags, lflags):
        self.name = name
        self.cflags = cflags
        self.lflags = lflags
        self.gdefines = [("HAVE_" + self.name.upper(), None)]

    def setup_module (self, module):
        if not module.canbuild:
            return

        # update compiler/linker args for the module
        module.cflags += self.cflags
        module.lflags += self.lflags
        module.globaldefines += self.gdefines 

def get_dependencies (cfg, compiler):
    dep = CompilerDependency
    deps = []
    if compiler == 'unix' or compiler == 'cygwin' or \
       compiler == 'mingw32' or compiler == 'mingw32-console':
        # TODO: check compiler version!
        if cfg.build['OPENMP']:
            deps.append (dep ('openmp', ['-fopenmp'], ['-fopenmp']))
    elif compiler == 'msvc':
        if cfg.build['OPENMP']:
            deps.append (dep ('openmp', ['/openmp'], ['/openmp']))
    elif compiler == 'icc':
        if cfg.build['OPENMP']:
            deps.append (dep ('openmp', ['-openmp'], ['-openmp']))

    for d in deps:
        print ("Adding compiler flags '%s':" % d.name)
        print ("")
        print ("\tCFlags : " + repr(d.cflags))
        print ("\tLFlags : " + repr(d.lflags))
        print ("")
        
    return deps
