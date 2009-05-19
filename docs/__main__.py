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
main_page = os.path.join(pkg_dir, 'index.html')
url_path = quote('/'.join(iterpath(main_page)))
url = urlunparse(('file', '', url_path, '', '', ''))
webbrowser.open(url)
