import bdist_mpkg
from distutils.command.bdist_mpkg import bdist_mpkg as _bdist_mpkg

FRAMEWORKS = ['SDL', 'SDL_ttf', 'SDL_image', 'SDL_mixer', 'smpeg']

class bdist_mpkg(_bdist_mpkg):
    def initialize_options(self):
        _bdist_mpkg.initialize_options(self)
        self.scheme_descriptions['examples'] = u'(Optional) pygame example code'
        self.scheme_map['examples'] = '/Developer/Python/pygame/Examples'
        self.scheme_copy['examples'] = 'examples'
        for framework in FRAMEWORKS:
            self.scheme_descriptions[framework] = u'(Required) %s.framework' % (framework,)
            self.scheme_map[framework] = '/Library/Frameworks/%s.framework' % (framework,)
            self.scheme_copy[framework] = '/Library/Frameworks/%s.framework' % (framework,)

cmdclass = {'bdist_mpkg': bdist_mpkg}
