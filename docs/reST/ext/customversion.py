from sphinx.domains.changeset import versionlabels, VersionChange
from sphinx.locale import _  # just to suppress warnings

try:
    from sphinx.domains.changeset import versionlabel_classes
except ImportError:
    # versionlabel_classes doesn't exist in old Sphinx versions.
    UPDATE_VERIONLABEL_CLASSES = False
else:
    UPDATE_VERIONLABEL_CLASSES = True


labels = ("versionadded", "versionchanged", "deprecated", "versionextended")


def set_version_formats(app, config):
    for label in labels:
        versionlabels[label] = _(getattr(config, f"{label}_format"))


def setup(app):
    app.add_directive("versionextended", VersionChange)
    versionlabels["versionextended"] = "Extended in pygame %s"

    if UPDATE_VERIONLABEL_CLASSES:
        versionlabel_classes["versionextended"] = "extended"

    for label in ("versionadded", "versionchanged", "deprecated", "versionextended"):
        app.add_config_value(
            f"{label}_format", str(versionlabels[label]), "env"
        )

    app.connect("config-inited", set_version_formats)
