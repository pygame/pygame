# only enable bdist_mpkg if it's installed already
import pkg_resources
try:
    pkg_resources.require('bdist_mpkg>=0.4.2')
except pkg_resources.DistributionNotFound:
    raise ImportError

FRAMEWORKS = ['SDL', 'SDL_ttf', 'SDL_image', 'SDL_mixer', 'smpeg']

CUSTOM_SCHEMES = dict(
    examples=dict(
        description=u'(Optional) pygame example code',
        prefix='/Developer/Python/pygame/Examples',
        source='examples',
    ),
    docs=dict(
        description=u'(Optional) pygame documentation',
        prefix='/Developer/Python/pygame/Documentation',
        source='docs',
    ),
)

for framework in FRAMEWORKS:
    CUSTOM_SCHEMES[framework] = dict(
        description=u'(Required) %s.framework' % (framework,),
        prefix='/Library/Frameworks/%s.framework' % (framework,),
        source='/Library/Frameworks/%s.framework' % (framework,),
    )

options = dict(bdist_mpkg=dict(custom_schemes=CUSTOM_SCHEMES))
