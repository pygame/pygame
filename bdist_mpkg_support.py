# only enable bdist_mpkg if it's installed already
import pkg_resources
try:
    pkg_resources.require('bdist_mpkg>=0.4.2')
except pkg_resources.DistributionNotFound:
    raise ImportError

try:
    unicode
except NameError:
    def unicode(s):
        return str(s)

#FRAMEWORKS = ['SDL', 'SDL_ttf', 'SDL_image', 'SDL_mixer', 'smpeg']
FRAMEWORKS = ['SDL', 'SDL_ttf', 'SDL_image', 'SDL_mixer']

CUSTOM_SCHEMES = dict(
    examples=dict(
        description=unicode('(Optional) pygame example code'),
        prefix=unicode('/Developer/Python/pygame/Examples'),
        source='examples',
    ),
    docs=dict(
        description=unicode('(Optional) pygame documentation'),
        prefix=unicode('/Developer/Python/pygame/Documentation'),
        source='docs',
    ),
)

for framework in FRAMEWORKS:
    CUSTOM_SCHEMES[framework] = dict(
        description=unicode('(Required) %s.framework' % (framework,)),
        prefix=unicode('/Library/Frameworks/%s.framework' % (framework,)),
        source=unicode('/Library/Frameworks/%s.framework' % (framework,)),
    )

options = dict(bdist_mpkg=dict(custom_schemes=CUSTOM_SCHEMES))
