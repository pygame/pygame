#!/usr/bin/env python

"""A one-shot Pygame DOC to reStructuredText translator.

This program is intended for making the move to reStructuredText for
Pygame document sources. Once done, the original sources will be retired
and future edits done in reST.

The program generates Python specific Sphinx 0.6 markup based on that used by jug.

An output directory path can be specified as a command line argument.
The default directory is reSTdoc. The directory is created if it
does not exist.

"""


import sys
import os
import glob
import re

WRAPLEN = 79
INDENT = 3

try:
    next
except NameError:
    def next(i):
        return i.next()

from collections import deque

TWORD = 0
TIDENT = 1
TFUNCALL = 2
TUPPER = 3
THYPERLINK = 4

class Getc(object):
    def __init__(self, s):
        self.i = iter(s)
        self.store = deque()

    def __call__(self):
        if self.store:
            return self.store.popleft()
        try:
            return next(self.i)
        except StopIteration:
            pass
        return ''

    def unget(self, s):
        self.store.extendleft(reversed(s))

break_chars = set(iter(',;:?!)}]'))

def tokenize(s):
    getc = Getc(s)
    token = deque()
    finish = ''
    loop = True
    while loop:
        ch = getc()
        if token:
            if not ch or ch.isspace():
                if len(token) == 1 and token[0] in 'AI':
                    ttype = TWORD
                elif alnum and hasdot and not hasargs:
                    ttype = TIDENT
                elif alnum and allupper and not hasargs:
                    ttype = TUPPER
                elif hasargs:
                    ttype = TFUNCALL
                elif hlink:
                    ttype = THYPERLINK
                else:
                    ttype = TWORD
                yield ttype, ''.join(token), finish
                token.clear()
                finish = ''
            elif ch == '.':
                next_ch = getc()
                if not next_ch or next_ch.isspace():
                    finish = ch
                else:
                    hasdot = (alnum and
                              (hasdot or next_ch.isalnum() or next_ch == '_'))
                    hlink = False
                    token.extend(ch)
                getc.unget(next_ch)
            elif ch.isalpha():
                allupper = allupper and ch.isupper()
                token.extend(ch)
                hlink = False
            elif ch.isdigit():
                token.extend(ch)
                hlink = False
            elif ch == '(' and alnum:
                getc.unget(ch)
                next_token = parens(getc)
                if next_token:
                    token.extend(next_token)
                    hasargs = alnum
                else:
                    alnum = False
                    allupper = False
                hlink = False
            elif ch in '_':
                token.extend(ch)
                hlink = False
            elif ch in '+_':
                token.extend(ch)
                alnum = False
                hlink = False
            elif ch in break_chars:
                next_ch = getc()
                if next_ch and not next_ch.isspace():
                    hasargs = False
                    allupper = False
                    alnum = False
                    token.extend(ch)
                else:
                    finish = ch
                getc.unget(next_ch)
            else:
                token.extend(ch)
                hasargs = False
                allupper = False
                alnum = False
                hlink = False
        elif ch.isspace():
            pass
        elif ch:
            alnum = False
            allupper = False
            hasdot = False
            hasargs = False
            hlink = False
            if ch == '_':
                alnum = True
                allupper = True
            elif ch.isalpha():
                alnum = True
                allupper = ch.isupper()
            elif ch == '*':
                ch = '\\*'
                alnum = True
                allupper = True
            elif ch == '`':
                ch = '\\`'
            elif ch == '<':
                getc.unget(ch)
                next_token = hyperlink(getc)
                if next_token:
                    token.extend(next_token)
                    hlink = True
            token.extend(ch)
        loop = bool(ch)

def parens(getc):
    ch = getc()
    token = deque()
    while ch:
        if ch == '(' and token:
            getc.unget(ch)
            subtoken = parens(getc)
            if subtoken:
                token.extend(subtoken)
            else:
                getc.unget(token)
                return ''
        else:
            token.append(ch)
            if ch == ')':
                return token
        ch = getc()
    return ''

class NotAMatch(StandardError):
    pass

def hyperlink(getc):
    token = deque()
    try:
        token.extend(token_match('<a ', getc))
        token.extend(token_find('>', getc))
        token.extend(token_find('<', getc, 200))
        token.extend(token_match('/a>', getc))
    except NotAMatch:
        getc.unget(token)
        return ''
    return token

def token_match(s, getc):
    token = deque()
    for c in s:
        ch = getc()
        if ch == c:
            token.append(ch)
        else:
            getc.unget(token)
            raise NotAMatch()
    return token

def token_find(c, getc, maxchar=100):
    token = deque()
    while maxchar:
        ch = getc()
        token.append(ch)
        if ch == c:
            return token
        maxchar -= 1
    getc.unget(token)
    raise NotAMatch()

def sortkey(x):
    return os.path.basename(x).lower()

def sort_list_by_keyfunc(alist, akey):
    """ sort(key=sortkey) is only in python2.4.
         this is not inplace like list.sort()
    """
    # make a list of tupples with the key as the first.
    keys_and_list = list( zip(map(akey, alist), alist) )
    keys_and_list.sort()
    alist = list( map(lambda x:x[1], keys_and_list) )
    return alist

def collect_doc_files():
    # ABSPATH ONLY WORKS FOR docs_as_dict
    #
    # if __name__ == '__main__': Run()
    #     must be ran from in trunk dir
    
    # get files and shuffle ordering
    trunk_dir = os.path.abspath(os.path.dirname(__file__))
    
    src_dir = os.path.join(trunk_dir, 'src')
    lib_dir = os.path.join(trunk_dir, 'lib')

    pygame_doc = os.path.join(src_dir, "pygame.doc")

    files = (
        glob.glob(os.path.join(src_dir,'*.doc')) +
        glob.glob(os.path.join(lib_dir,'*.doc'))
    )

    files.remove(pygame_doc)
    
    #XXX: sort(key=) is only available in >= python2.4
    # files.sort(key=sortkey)
    files = sort_list_by_keyfunc(files, sortkey)

    files.insert(0, pygame_doc)

    return files

def Run(target_dir=None):
    global roles

    if target_dir is None:
        target_dir = 'reSTdoc'
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    files = collect_doc_files()
    for file in files:
        # print file
        print (os.path.basename(file))
    
    docs = []
    pages = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        pages.append(name)
        d = name, Doc('', open(f, "U"))
        docs.append(d)
    
    #pages.sort(key=str.lower)
    pages = sort_list_by_keyfunc(pages, str.lower)

    pages.insert(0, "index")
    
    index = {}
    justDocs = []
    for name, doc in docs:
        justDocs.append(doc)
        MakeIndex(name, doc, index)

    for name, doc in docs:
        fullname = os.path.join(target_dir, "%s.rst") % name
        outFile = open(fullname, "w")
        WriteHeader(outFile, doc)
        reSTOut(doc, index, outFile)
        outFile.close()


def MakeIndex(name, doc, index, level=0):
    if doc.fullname and '.' in doc.fullname[1:-1]:
        if level == 0:
            markup = ":mod:`%s`" % doc.fullname
        elif doc.kids:
            markup = ":class:`%s`" % doc.fullname
        elif doc.protos and level == 1:
            markup = ":func:`%s`" % doc.fullname
        elif doc.protos:
            markup = ":meth:`%s`" % doc.fullname
        elif level == 1:
            markup = ":data:`%s`" % doc.fullname
        else:
            markup = ":attr:`%s`" % doc.fullname
        index[doc.fullname] = markup
    if doc.kids:
        for kid in doc.kids:
            MakeIndex(name, kid, index, level+1)

def imapcdr(fn, seq):
    while seq:
        yield fn(seq)
        seq = seq[1:]

proto_pat = re.compile(r'[a-zA-Z_][a-zA-Z0-9_.]* *\((?:\([^)]*\)|[^()]*)*\)')
return_pat = re.compile(r'(?:(?: +[rR]eturns? *)|(?: *[-=]> *))(?P<ret>.+)')
name_pat = re.compile(r'\.?(?P<name>[a-zA-Z_][a-zA-Z0-9_.]*)')
value_pat = re.compile(r'[a-zA-Z_][a-zA-Z0-9_.]* += +(?P<value>.+)')

def strip_modname(name, modname):
    for submodname in imapcdr('.'.join, modname.split('.')):
        if name.startswith(submodname):
            return name[len(submodname) + 1:]
    return name

def pstrip(proto, modname):
    n = ''
    p = ''
    r = ''
    v = ''

    proto = strip_modname(proto, modname)
    m = name_pat.match(proto)
    if m is not None:
        n = m.group('name')
    m = proto_pat.search(proto)
    if m is not None:
        p = m.group(0)
        m = return_pat.search(proto)
        if m is not None:
            r = m.group('ret')
            print ('modname: %s, n: %s, p: %s, ret: %s' %
                   (modname, n, p, r))  ##
        else:
            r = 'None'
        return n, p, r, v
    if n:
        m = return_pat.search(proto)
        if m is not None:
            r = m.group('ret')
        else:
            m = value_pat.search(proto)
            if m is not None:
                v = m.group('value')
        return n, p, r, v
    print ("Unknown: %s" % proto)
    return n, p, r, v

def ipstrip(protos, modname):
    for proto in protos:
        yield pstrip(proto, modname)

def reSTOut(doc, index, f, level=0, new_module=True, modname=''):
    indent = ' ' * (INDENT * level)
    content_indent = ' ' * (INDENT * (level + 1))
    descr = doc.descr.strip()
    desctype = None
    protos = None
    name = strip_modname(doc.fullname, modname)

    if new_module:
        WriteHeading(f, doc)

    if new_module and (level > 0 or not doc.protos):
        level = 0
        indent = ' ' * (INDENT * level)
        content_indent = ' ' * (INDENT * (level + 1))
        modname = doc.fullname
        f.write('\n')
        f.write('%s.. module:: %s\n' % (indent, modname))
        f.write('%s:synopsis: %s\n' % (content_indent, descr))
    elif doc.protos:
        if new_module:
            modname = '.'.join(doc.fullname.split('.')[:-1])
            name = doc.fullname.split('.')[-1]
            f.write('\n')
            f.write('%s.. currentmodule:: %s\n' % (indent, modname))
        n, p, r, v = pstrip(doc.protos[0], modname)
        if doc.protos[0].startswith('raise '):
            desctype = 'exception'
            protos = doc.protos
            modname += '.' + name
        elif not p and n and r and level > 0:
            desctype = 'attribute'
            protos = ['%s -> %s' % (n, r) for n, p, r, v in ipstrip(doc.protos, modname)]
        elif not p and v:
            desctype = 'data'
            protos = ['%s = %s' % (n, v) for n, p, r, v in ipstrip(doc.protos, modname)]
        elif not p and doc.kids:
            f.write('\n')
            reSTOut(doc, index, f, level + 1)
            return
        elif not p:
            desctype = 'describe'
            protos = doc.protos
        elif doc.kids:
            desctype = 'class'
            protos = []
            for proto in doc.protos:
                n, p, r, v = pstrip(proto, modname)
                if not r:
                    r = n
                protos.append('%s -> %s' % (p, r))
            modname += '.' + name
        else:
            if level == 0:
                desctype = 'function'
            else:
                desctype = 'method'
            protos = ['%s -> %s' % (p, r) for n, p, r, v in ipstrip(doc.protos, modname)]
        level += 1
    elif doc.fullname:
        desctype = 'data'
        level += 1

    if desctype is not None:
        f.write('\n')
        f.write('%s.. %s:: %s\n' % (indent, desctype, name))

    indent = ' ' * (INDENT * level)
    content_indent = ' ' * (INDENT * (level + 1))

    if descr or protos is not None:
        f.write('\n')

    if descr:
        f.write('%s| :sl:`%s`\n' % (indent, descr))
        
    if protos is not None:
        for proto in protos:
            f.write('%s| :sg:`%s`\n' % (indent, proto))

    if doc.docs:
        pre = False
        for d in doc.docs:
            if d[0] == '*':
                for li in d[1:].split('*'):
                    txt = reSTPrettyWord(li)
                    f.write('\n')
                    f.write("%s* %s\n" % (content_indent, txt))
            else:
                txt, pre = reSTPrettyLine(d, index, pre, level)
                f.write(txt)

    if doc.kids:
        for k in doc.kids:
            reSTOut(k, index, f, level, False, modname)

    f.write('\n')
    f.write("%s.. ## %s ##\n" % (indent, doc.fullname))


def reSTPrettyWord(word):
    parts = []
    for ttype, token, finish in tokenize(word):
        if ttype == TWORD:
            parts.append(token + finish)
        else:
            parts.append('``%s``%s' % (token, finish))
    return ' '.join(parts)


def reSTPrettyLine(line, index, pre, level):
    pretty = ""

    indent = ' ' * (INDENT * level)

    line = apply_markup(line)
    
    if line[0].isspace():
        if not pre:
            pretty += "\n%s::\n\n" % indent
            pre = True
    elif pre:
        pre = False

    if not pre:
        pretty += '\n'
        pretty += indent
        spacer = ''
        linelen = len(pretty) - 1
        for ttype, word, finish in tokenize(line):
            link = index.get(word)
            if link:
                markup = "%s%s" % (link, finish)
            elif ttype == TIDENT or ttype == TFUNCALL:
                markup = "``%s``%s" % (word, finish)
            elif ttype == TUPPER and (len(word) > 1 or word not in ['A', 'I']):
                markup = "``%s``%s" % (word, finish)
            elif ttype == THYPERLINK:
                try:
                    uri, reference = parse_hyperlink(word)
                except ValueError:
                    markup = "%s%s" % (word, finish)
                else:
                    markup = "`%s <%s>`_" % (reference, uri)
            else:
                markup = "%s%s" % (word, finish)
            mlen = len(markup)
            if linelen + mlen < WRAPLEN:
                pretty += spacer + markup
                linelen += len(spacer) + mlen
            else:
                pretty += '\n' + indent + markup
                linelen = len(indent) + mlen
            spacer = ' '
        pretty += '\n'
    else:
        pretty += indent + line + '\n'
    return pretty, pre

hyperlink_pat = re.compile(r'<a.*href="(?P<uri>.+)"[^>]*>(?P<ref>[^<]*)</a>')

def parse_hyperlink(token):
    m = hyperlink_pat.match(token)
    if m is None:
        raise ValueError("Token not html markup")
    return m.group('uri'), m.group('ref')

markup_pat = re.compile("&[a-z]+;")
markup_table = {'&lt;': '<', '&gt;': '>'}

def apply_markup(text):
    def repl(m):
        markup = m.group(0)
        return markup_table.get(markup, markup)
    return markup_pat.sub(repl, text)

def WriteHeader(outFile, doc):
    outFile.write('.. include:: common.txt\n')
    outFile.write('\n')

def WriteHeading(outFile, doc):
    v = vars(doc)
    title = (':mod:`%(fullname)s`' % v)
    outFile.write(title)
    outFile.write('\n')
    outFile.write('=' * len(title))
    outFile.write('\n')

def LayoutDocs(docs):
    levels = {}
    for doc in docs:
        if doc.fullname:
            topName = doc.fullname.split(".")[-1]
            levels[topName] = doc

    top = levels["pygame"]
    for doc in docs:
        if doc is top:
            continue
        #print (doc)
        if doc.fullname:
            parentName = doc.fullname.split(".")[-2]
        else:
            parentName = ""
        parent = levels.get(parentName)
        if parent is not None:
            parent.kids.append(doc)

    return top


class Doc(object):
    def __init__(self, parentname, f):
        self.kids = None
        self.protos = []
        self.docs = None
        self.descr = ""
        self.name = ""
        self.fullname = ""
        self.finished = False

        curdocline = ""
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip()

            if line == "<END>":
                if curdocline:
                    self.docs.append(curdocline)
                    curdocline = ""
                self.finished = True
                break

            if self.kids is not None:
                kid = Doc(self.fullname, f)
                if kid:
                    self.kids.append(kid)


            if line == "<SECTION>":
                if curdocline:
                    self.docs.append(curdocline)
                    curdocline = ""
                self.kids = []
                continue
            
            if line:
                if self.docs is not None:
                    if line[0].isspace():
                        if curdocline:
                            self.docs.append(curdocline)
                            curdocline = ""
                        self.docs.append(line)
                    else:
                        curdocline += line + " "
                elif not self.name:
                    self.name = line
                    if len(line) > 1 and line[0] == '"' and line[-1] == '"':
                        self.fullname = line[1:-1]
                    elif parentname:
                        splitparent = parentname.split(".")
                        if splitparent[-1][0].isupper():
                            self.fullname = splitparent[-1] + "." + line
                        else:
                            self.fullname = parentname + "." + line
                    else:
                        self.fullname = line
                elif not self.descr:
                    self.descr = line
                else:
                    self.protos.append(line)
            else:
                if self.docs is not None:
                    if curdocline:
                        self.docs.append(curdocline)
                    curdocline = ""
                elif self.name and self.kids is  None:
                    self.docs = []

    def __repr__(self):
        return "<Doc '%s'>" % self.name
            
    def __nonzero__(self):
        return self.finished

    def __cmp__(self, b):
        return cmp(self.name.lower(), b.name.lower())

def docs_as_dict():
    """

    Dict Format:

        {'pygame.rect.Rect.center': 'Rect.center: ...' ...}

    Generally works, has some workarounds, inspect results manually.

    """

    import pygame
    files = collect_doc_files()

    def make_mapping(doc, parent_name):
        docs = {}
        for k in doc.kids:
            if k.docs:
                kid_name = k.fullname

                if parent_name == 'pygame':
                    if hasattr(pygame.base, k.name):
                        kid_name = '%s.%s' % ('pygame.base', k.name)

                elif not kid_name.startswith(parent_name):
                    kid_name = '%s.%s' % (parent_name, kid_name)

                docs[kid_name] = '\n'.join(k.docs)

            if k.kids:
                docs.update(make_mapping(k, parent_name))
        return docs

    mapping = {}
    for f in files:
        doc = Doc('', open(f, "U"))
        mapping.update(make_mapping(doc, doc.name.lower()))

    return mapping


reSTMid = ''

reSTFinish = ''

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = None
    Run(target_dir)
