#!/usr/bin/env python

"""A one-shot Pygame DOC to reStructuredText translator.

This program is intended for making the move to reStructuredText for
Pygame document sources. Once done, the original sources will be retired
and future edits done in reST.

Right now, the program generates Python specific markup as implemented
by jug.

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
TUPPER = 2

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


def tokenize(s):
    getc = Getc(s)
    token = deque()
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
                else:
                    ttype = TWORD
                yield ttype, ''.join(token)
                token.clear()
            elif ch == '.':
                next_ch = getc()
                hasdot = (alnum and
                          (hasdot or next_ch.isalnum() or next_ch == '_'))
                getc.unget(next_ch)
                token.extend(ch)
            elif ch.isalnum():
                allupper = allupper and ch.isupper()
                token.extend(ch)
            elif ch == '(' and alnum:
                getc.unget(ch)
                next_token = parens(getc)
                if next_token:
                    token.extend(next_token)
                    hasargs = alnum
                else:
                    alnum = False
                    allupper = False
            elif ch in '_':
                token.extend(ch)
            elif ch in '+_':
                alnum = False
                token.extend(ch)
            else:
                token.extend(ch)
                hasargs = False
                allupper = False
                alnum = False
        elif ch.isspace():
            pass
        elif ch:
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
                alnum = False
                allupper = False
            else:
                alnum = False
                allupper = False
            token.extend(ch)
            hasdot = False
            hasargs = False
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
        WritePageLinks(outFile, pages)
        outFile.write(reSTMid)
        reSTOut(doc, index, outFile)
        outFile.write(reSTFinish)
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

proto_pat = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]* *\((?:\([^)]*\)|[^()]*)*\)')
return_pat = re.compile(r'(?:(?: +[rR]eturns? *)|(?: *[-=]> *))(?P<ret>.+)')
name_pat = re.compile(r'\.(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(:| =|\n)')
value_pat = re.compile(r'\.[a-zA-Z_][a-zA-Z0-9_]* += +(?P<value>.+)\n')

def pstrip(proto):
    p = ''
    r = ''
    v = ''
    m = proto_pat.search(proto)
    if m is not None:
        p = m.group(0)
        m = return_pat.search(proto)
        if m is not None:
            r = m.group('ret')
            print ('p: %s, ret: %s' % (p, r))  ##
        else:
            r = 'None'
        return p, r, v
    m = name_pat.search(proto)
    if m is not None:
        p = m.group('name')
        m = return_pat.search(proto)
        if m is not None:
            r = m.group('ret')
        else:
            m = value_pat.search(proto)
            if m is not None:
                v = m.group('value')
        return p, r, v
    print ("Unknown: %s" % proto)
    return p, r, v


def reSTOut(doc, index, f, level=0, prefix=''):
    f.write('\n\n')
    if level == 0:
        indent = ''
        f.write('.. module:: %s\n' % doc.fullname)
        f.write('%s:synopsis: %s\n' % ((' ' * INDENT), doc.descr))
    else:
        indent = ' ' * (INDENT * (level - 1))
        content_indent = ' ' * (INDENT * level)
        if doc.protos:
            p, r, v = pstrip(doc.protos[0])
            if not p:
                f.write('%s.. describe:: %s\n' % (indent, doc.fullname))
                f.write('\n')
                f.write('%s.. FIXME: Needs hand formatting\n' %
                        (content_indent,))
                f.write('\n')
                for proto in doc.protos:
                    f.write('%s| **%s**\n' % (content_indent, proto))
            elif doc.kids:
                f.write('%s.. class:: %s\n' % (indent, p))
                for proto in doc.protos[1:]:
                    p, r, v = pstrip(proto)
                    f.write('%s           %s\n' % (indent, p))
            elif p.endswith(')') and level > 1:
                f.write('%s.. method:: %s -> %s\n' % (indent, p, r))
                for proto in doc.protos[1:]:
                    p, r, v = pstrip(proto)
                    f.write('%s            %s -> %s\n' % (indent, p, r))
                f.write('\n')
            elif level > 1:
                f.write('%s.. attribute:: %s\n' % (indent, p))
                if r and v:
                    f.write('\n')
                    f.write('%s.. FIXME: both a value and a return?\n')
                if r:
                    f.write('\n')
                    f.write('%sreturn: **%s**\n' % (content_indent, r))
                if v:
                    f.write('\n')
                    f.write('%svalue: **%s**\n' % (content_indent, v))
            else:
                f.write('%s.. function:: %s -> %s\n' % (indent, p, r))
                for proto in doc.protos[1:]:
                    p, r, v = pstrip(proto)
                    f.write('%s              %s -> %s\n' % (indent, p, r))
            f.write('\n')
            f.write('%s.. FULLNAME: %s\n' % (content_indent, doc.fullname))
        elif doc.fullname:
            f.write('%s.. data:: %s\n' % (content_indent, doc.fullname))

        if doc.descr:
            f.write('\n')
            f.write('%s**%s**\n' % (content_indent, doc.descr))

        indent = content_indent

    if doc.docs:
        f.write('\n')
        pre = False
        for d in doc.docs:
            if d[0] == '*':
                f.write('\n')
                for li in d[1:].split('*'):
                    txt = reSTPrettyWord(li)
                    f.write("%s- %s\n" % (indent, txt))
            else:
                txt, pre = reSTPrettyLine(d, index, pre, level)
                f.write(txt)

    level += 1

    if doc.kids:
        for k in doc.kids:
            reSTOut(k, index, f, level, prefix)

    f.write('\n')
    f.write("%s.. ## %s ##\n" % (indent, doc.fullname))


def reSTPrettyWord(word):
    if '.' in word[:-1] or word.isupper():
        return word
    return word


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
        for ttype, word in tokenize(line):
            if word[-1] in ",.":
                finish = word[-1]
                word = word[:-1]
            else:
                finish = ""
            link = index.get(word)
            if link:
                markup = "%s%s" % (link, finish)
            elif ttype == TIDENT or ttype == TFUNCALL:
                markup = "``%s``%s" % (word, finish)
            elif ttype == TUPPER and (len(word) > 1 or word not in 'AI'):
                markup = "``%s``%s" % (word, finish)
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

markup_pat = re.compile("&[a-z]+;")
markup_table = {'&lt;': '<', '&gt;': '>'}

def apply_markup(text):
    def repl(m):
        markup = m.group(0)
        return markup_table.get(markup, markup)
    return markup_pat.sub(repl, text)

def WriteHeader(outFile, doc):
    v = vars(doc)
    title = (':mod:`%(fullname)s`' % v)
    outFile.write(title)
    outFile.write('\n')
    outFile.write('=' * len(title))
    outFile.write('\n')


def WritePageLinks(outFile, pages):
    return   ##
    outFile.write('.. # The following was generated by WritePageLinks.\n\n')
    outFile.write('|| `Pyame Home <http://www.pygame.org>`_ '
                  '|| `Help Contents <../index.html>`_'
	          '|| `Reference Index <index.html>`_'
	          '||\n\n')
    links = []
    for page in pages[1:]:
        link = '`%s <%s.html>`_' % (page, page.title())
        links.append(link)
    outFile.write(' || '.join(links))
    outFile.write('\n\n')


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
