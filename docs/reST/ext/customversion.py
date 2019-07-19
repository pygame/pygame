from sphinx.domains.changeset import versionlabels, VersionChange
from sphinx.locale import _ # just to suppress warnings


labels = ('versionadded', 'versionchanged', 'deprecated', 'versionextended')


def set_version_formats(app, config):
    for label in labels:
        versionlabels[label] = \
            _(getattr(config, '{}_format'.format(label)))


def setup(app):
    app.add_directive('versionextended', VersionChange)
    versionlabels['versionextended'] = 'Extended in pygame %s'

    for label in ('versionadded', 'versionchanged', 'deprecated', 'versionextended'):
        app.add_config_value('{}_format'.format(label), str(versionlabels[label]), 'env')

    app.connect('config-inited', set_version_formats)
