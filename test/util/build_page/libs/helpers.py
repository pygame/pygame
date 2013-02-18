################################################################################

import re
import zipfile

from os.path import normpath, join, dirname, abspath

import safe_eval

from build_client.helpers import create_zip

from pywebsite.escape import ehtml

def relative_to(f, rel):
    return normpath(join(abspath(dirname(f)), rel)).strip()

def slug(s):
    s = s.strip().lower()
    s = re.sub('[^a-z0-9-]', '-', s)
    return re.sub('-+', '-', s)

def norm_le(s):
    return re.sub('\r\n|\r', '\n', s)

################################################################################

class ResultsZip(zipfile.ZipFile):
    def __init__(self, *args, **kw):
        zipfile.ZipFile.__init__(self, *args, **kw)
        self.config   = self.eval('config.txt')
        self.text_files = set(t for t in self.namelist() if t != self.installer)

    @property
    def installer(self):
        if 'prebuilt' in self.namelist():
            return norm_le(self.read('prebuilt')).split('\n')[2].strip()

    def eval(self, key):
        data = norm_le(self.read(key))
        return safe_eval.safe_eval(data)

    def __getattr__(self, attr):
        return self.config[attr]

    def archive_text(self, path):
        create_zip(path, **dict((k, self.read(k)) for k in self.text_files))

    def html(self, the_main):
        all_tabs = []
        slugs = []

        for i, f in enumerate(sorted(self.text_files)):
            fslug = slug(f)

            all_tabs.append (
                "<li class='t%(i)s'><a class='t%(i)s tab' href='%(fslug)s'>"
                "%(f)s</a></li>" % locals()
            )

            if fslug == the_main:
                contents = ehtml(self.read(f), 'utf-8')
                contents = "<div class='t%s'><pre>%s</pre></div>" % (i, contents)

        return '\n'.join(all_tabs), contents
        
################################################################################

__all__ = ['relative_to', 'slug', 'ResultsZip']

################################################################################

if __name__ == '__main__':
    rz = ResultsZip('results.zip')
    rz.printdir()
    print rz.platform_id
    rz.dump_file('bla.zip')