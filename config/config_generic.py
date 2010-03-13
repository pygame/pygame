import os, glob

class Dependency (object):
    """
    Generic Dependency class.
    
    This class represents a generic library dependency of a PyGame
    module, i.e. the required compiler and linker flags for building
    a module which depends on this library.
    
    This class is instantiated with the 'linker id' (for example, a
    library which is linked with '-lpng' has 'png' as its linker id)
    of the library it represents and the name of a representative
    header file.
    
    It must be then configured with the configure() method; if the
    configuration is successful, the setup_module() method may be
    used to update a Module object with the compiler flags which are
    needed to build it.
    
    The process of 'configuring' this Dependency implies setting the
    following lists with the information relevant to the library:
    
        self.incdirs  (all the directories which contain the include files)
        self.libdirs  (the directory(s) which contain the library itself)
        self.libs     (the name of the library files which must be linked)
        self.cflags   (all the flags which must be passed to the C compiler)
        self.gdefines (all the defines which to be passed to the C compiler)
        self.lflags   (all the flags which must be passed to the C linker)

    Configuration is done by executing 'configuration callbacks'
    (each one implementing a different configuration method) until
    the required information has been collected.
    
    A configuration callback is simply a class method starting by
    '_configure_' which is automatically executed by the
    Dependency.configure() method and returns True if the library's
    information was found.
    
    By default, this generic class implements the configuration
    method which is common to all platforms:
    Dependency._configure_guess() tries to guess the location of the
    library by looking in system folders for the required files.
    
    However, all platforms inherit this class to implement their
    custom configuration callbacks; for instance, Unix systems also
    try to use the 'pkgconfig' tool to automatically locate
    installed libraries, and Mac OS X tries to locate installed
    Framework Bundles containing the libraries.
    
    See the other configuration callbacks in their respective modules:
    
        config.config_unix
        config.config_darwin
        config.config_msys
        config.config_win
    """

    _incdirs = []
    _libdirs = []
    _searchdirs = []
    _libprefix = ""

    def __init__(self, header_files, library_link_id,
            config_program = None, pkgconfig_name = None,
            extra_include_dirs = []):

        self.header_files = header_files

        self.library_name = self._libprefix + library_link_id
        self.library_id = library_link_id
        self.library_config_program = config_program
        self.pkgconfig_name = pkgconfig_name

        self.incdirs = [] + extra_include_dirs
        self.libdirs = []
        self.libs = [self.library_id]
        self.cflags = []
        self.lflags = []
        self.gdefines = []
        self.nocheck = False

        self.configured = False

    def _canbuild(self, cfg):
        """
        Returns if this library has been manually disabled
        by the user on the 'cfg.py' module, and hence cannot be built.
        """
        cfg_entry = self.library_id.upper()

        if cfg_entry in cfg.build:
            return cfg.build[cfg_entry]

        # if the library doesn't show up on cfg.py, we assume that the user
        # wants to actually compile it.
        return True

    def _find_libdir(self, name):
        """
        Searches the library folders for the specified library
        file.
        """
        for d in self._searchdirs:
            for g in self._libdirs:
                p = os.path.join (d, g)
                gl = glob.glob(os.path.join(p, name) + '*')

                for f in gl:
                    if os.path.isfile(f): return p

        return None

    def _find_incdir(self, name):
        """
        Recursively search all include dirs for the specified
        header file.
        """
        for d in self._searchdirs:
            for g in self._incdirs:
                path = os.path.join(d, g)
                for (path, dirnames, filenames) in os.walk(path):
                    if name in filenames:
                        return path

        return None

    def _configure_guess(self):
        """
        Configuration callback which automatically looks for
        the required headers and libraries in some default
        system folders.
        """
        dirs = []
        for h in self.header_files:
            directory = self._find_incdir (h)
            if directory is None:
                return False
            dirs.append (directory)
        
        libdir = self._find_libdir (self.library_name)
        if libdir is None:
            return False

        self.incdirs.extend(dirs)
        self.libdirs.append(libdir)
        return True

    _configure_guess.priority = 0

    def configure(self, cfg):
        """
        Finds the compiler/linker information needed
        to configure this library.
        """
        print ("Configuring library '%s':" % self.library_id)
        if self.nocheck:
            # Avoid any check and assume, the dependency is already in shape.
            self.configured = True
            print ("Configuration not needed...")
            self.gdefines.append(("HAVE_" + self.library_id.upper(), None))
            print ("")
            print ("\tCFlags : " + repr(self.cflags))
            print ("\tLFlags : " + repr(self.lflags))
            print ("\tIncDirs: " + repr(self.incdirs))
            print ("\tLibDirs: " + repr(self.libdirs))
            print ("\tLibs   : " + repr(self.libs))
            print ("")
            return

        if not self._canbuild(cfg):
            print ("\tLibrary '%s' has been manually disabled.\n" % self.library_id)
            return


        # find all configuration callbacks
        configure_callbacks = [ getattr(self, attr)
                                for attr in dir(self)
                                if attr.startswith('_configure_') ]
        
        # helper method for sort
        def _get_priority(cb):
            if hasattr(cb, 'priority'):
                return cb.priority
            return 0

        # sort them by priority; callbacks without a priority attribute
        # default to 0
        configure_callbacks.sort(reverse = True, key = _get_priority)

        for callback in configure_callbacks:
            callback_name = callback.__name__[11:].title()

            if callback():
                print (("Attempting to configure with %s..." % callback_name).ljust(50)
                        + "Success!")
                self.configured = True

                self.cflags = list(set(self.cflags))
                self.lflags = list(set(self.lflags))
                self.incdirs = list(set(self.incdirs))
                self.libdirs = list(set(self.libdirs))
                self.libs = list(set(self.libs))
                self.gdefines.append(("HAVE_" + self.library_id.upper(), None))

                print ("")
                print ("\tCFlags : " + repr(self.cflags))
                print ("\tLFlags : " + repr(self.lflags))
                print ("\tIncDirs: " + repr(self.incdirs))
                print ("\tLibDirs: " + repr(self.libdirs))
                print ("\tLibs   : " + repr(self.libs))
                print ("")

                # once the library has been configured with one configuration
                # callback, stop trying to configure it again
                return
                
            print (("Attempting to configure with %s..." % callback_name).ljust(50) 
                    + "Failure.")

        print ("\tFailed to configure library %s.\n" % self.library_id)

    def setup_module(self, module, optional = False):
        """
        Updates a modules.Module object with all the compiler
        and linker flags required to build it with this library.
        
        module - modules.Module object
        optional - whether the module requires the library to build
        """

        # if the building for this module is disabled already, it means
        # one of the other library prerequisites couldn't be configured
        # and the module won't be built no matter what
        if not module.canbuild:
            return

        # if this library hasn't been able to be configured, and it's
        # required for the module to build, we have to disable the building
        # of such module
        if not self.configured:
            module.canbuild = optional
            return

        # update compiler/linker args for the module
        module.cflags += self.cflags
        module.lflags += self.lflags
        module.incdirs += self.incdirs
        module.libdirs += self.libdirs
        module.globaldefines += self.gdefines 
        module.libs += self.libs
