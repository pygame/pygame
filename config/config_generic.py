import os

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
            
            self.incdirs (all the directories which contain the include files)
            self.libdirs (the directory(s) which contain the library itself)
            self.libs    (the name of the library files which must be linked)
            self.cflags  (all the flags which must be passed to the C compiler)
            self.lflags  (all the flags which must be passed to the C linker)

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

    def __init__(self, header_file, library_link_id):
        self.header_file = header_file
        self.library_name = 'lib' + library_link_id
        self.library_id = library_link_id

        self.incdirs = []
        self.libdirs = []
        self.libs = [self.library_id]
        self.cflags = []
        self.lflags = []

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

    def _configure_guess(self):
        """
            Configuration callback which automatically looks for
            the required headers and libraries in some default
            system folders.
        """
        directory = self._find_incdir(self.header_file)

        if directory is None:
            return False

        self.incdirs.append(directory)
        self.libdirs.append(self._find_libdir(self.library_name))
        self.cflags.append("-DHAVE_" + self.library_id.upper())
        return True

    _configure_guess.priority = 0

    def configure(self, cfg):
        """
            Finds the compiler/linker information needed
            to configure this library.
        """
        print ("Configuring library '%s':" % self.library_id)

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
                print (("Attempting to configure with %s..." % callback_name).ljust(50) + "Success!")
                self.configured = True

                self.cflags = list(set(self.cflags))
                self.lflags = list(set(self.lflags))
                self.incdirs = list(set(self.incdirs))
                self.libdirs = list(set(self.libdirs))
                self.libs = list(set(self.libs))

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
                
            print (("Attempting to configure with %s..." % callback_name).ljust(50) + "Failure.")

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
        module.cflags += list(self.cflags)
        module.lflags += list(self.lflags)
        module.incdirs += list(self.incdirs)
        module.libdirs += list(self.libdirs)
        module.libs += list(self.libs)


class DependencySDL (Dependency):
    """
        Generic SDL Dependency class.

        This class represents a SDL-based library which must be linked with
        a module. The main difference between a normal dependency and a SDL
        dependency is that the later must always contain our custom 'src/sdl'
        include dir.

        Additionally, some platforms implement custom SDL configuration callbacks: 
            Unix and Msys use the 'sdl-config' utility and Windows uses some special
            handling when linking SDL libraries.

        See: 
            config.config_unix.DependencySDL._configure_sdlconfig
            config.config_win.DependencySDL._configure_guess
    """

    def __init__(self, header_file, library_id):
        Dependency.__init__(self, header_file, library_id)

        # custom SDL include directory
        self.incdirs.append(os.path.join ("src", "sdl"))

