# python -m pygame.docs

import os
import webbrowser

try:
    from urllib.parse import urlunparse, quote
except ImportError:
    from urlparse import urlunparse
    from urllib import quote


def iterpath(path):
    path, last = os.path.split(path)
    if last:
        for p in iterpath(path):
            yield p
        yield last


pkg_dir = os.path.dirname(os.path.abspath(__file__))
main_page = os.path.join(pkg_dir, 'generated', 'index.html')
if os.path.exists(main_page):
    url_path = quote('/'.join(iterpath(main_page)))
    drive, rest = os.path.splitdrive(__file__)
    if drive:
        url_path = "%s/%s" % (drive, url_path)
    url = urlunparse(('file', '', url_path, '', '', ''))
else:
    url = "https://www.pygame.org/docs/"
webbrowser.open(url)
