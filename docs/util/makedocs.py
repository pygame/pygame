"""take all the PyGame source and create html documentation"""

import fnmatch, glob, os, types


DOCPATTERN = '*static char*doc_*=*'

SOURCES = ['../../src/*.c']
IGNORE_SOURCES = ['rwobject.c']
PYTHONSRC = ['color', 'cursors', 'version', 'sprite', 'sysfont']

OUTPUTDIR = '../ref/'
PAGETEMPLATE = open('pagelate.html').readlines()
DOCTEMPLATE = open('doclate.html').readlines()
LISTTEMPLATE = open('listlate.html').readlines()
INDEXTEMPLATE = ['<a href=ref/{category}.html#{name}>{mod}.{name}</a> - {quick}<br>']

INDEXSTART = "\n<br><hr><br><font size=+1><b>Full Index</b></font><ul>\n<!--FULLINDEX-->\n"
INDEXEND = "<!--ENDINDEX-->\n</ul>\n"

MODULETOC = ""


mainindex_desc = """
<tr><td>
The <b>pygame documentation</b> is mainly generated automatically from the
source code. Each module and object in the package is broken into its
own page in the reference documentation. The names of the objects are
capitalized, while the regular module names are lower case.
<br>&nbsp;<br>
The <b>pygame documentation</b> also comes with a full set of tutorials.
You can find links to these tutorials and other documentation files below.
</td></tr>

"""



def filltemplate(template, info):
    output = []
    for line in template:
        line = line.rstrip()
        pos = 0
        while 1:
            pos = line.find('{', pos)
            if pos == -1: break
            end = line.find('}', pos)
            if end == -1:
                pos += 1
                continue
            lookname = line[pos+1:end]
            match = info.get(lookname, '')
            if not match:
                pos = end
                continue
            try:line = line[:pos] + match + line[end+1:]
            except:
                print lookname, type(match), match
                raise
            pos += len(match) - (end-pos)
        output.append(line)
    return '\n'.join(output) + '\n'



def readsource(filename):
    documents = []
    file = open(filename)
    #read source
    while 1:
        line = file.readline()
        if not line: break
        if fnmatch.fnmatch(line, DOCPATTERN) and line.find(';') == -1:
            lines = [line]
            while 1:
                line = file.readline()
                if line.find("NODOC") >= 0 or line[0]=='#': continue
                if not line or line.find('"') == -1: break
                line = line[line.find('"')+1:line.rfind('"')]
                line = line.rstrip()
                if line == '\\n':
                    line = '<br>&nbsp;<br>'
                elif line.endswith('\\n'):
                    line = line[:-2]
                lines.append(line)
            documents.append(lines)
    return documents


def getpydoclines(doc):
    if doc is None: return
    lines = []
    for line in doc.split('\n'):
        if line == '':
            line = '<br>&nbsp;<br>'
        lines.append(line)
    return lines


def readpysource(name):
    modulename = 'pygame.' + name
    documents = []
    module = getattr(__import__(modulename), name)
    title = '    /*DOC*/ static char doc_pygame_' + name + '_MODULE[] =\n'
    documents.append([title] + getpydoclines(module.__doc__))
    modname = name
    for name, obj in module.__dict__.items():
        if type(obj) in (types.ClassType, types.TypeType):
            title = '    /*DOC*/ static char doc_%s[] =\n'%(name)
            initdocs = []
            if hasattr(obj, '__init__'):
                    init = getattr(obj, '__init__')
                    if hasattr(init, '__doc__'):
                            initdocs = getpydoclines(init.__doc__)
                            if not initdocs: initdocs = []
            try:
                docs = getpydoclines(obj.__doc__)
                if docs:
                    quick = '<b>(class)</b> - ' + docs[0]
                    usage = 'pygame.%s.%s()'%(modname,name)
                    if initdocs:
                        usage = 'pygame.%s.%s'%(modname,name) + initdocs[0][initdocs[0].find('('):]
                    docs = [usage,  quick,  '']+docs+['<br>&nbsp;<br>']+initdocs[2:]
                    documents.append([title] + docs)

            except AttributeError:
                documents.append([title] + ['%s.%s'&(modname,name),'noclassdocs'])
            for methname, meth in obj.__dict__.items():
                if methname[0] == '_': continue
                if type(meth)  is not types.FunctionType: continue
                title = '    /*DOC*/ static char doc_%s_%s[] =\n'%(name,methname)
                try:
                    docs = getpydoclines(meth.__doc__)
                    if docs:
                        docs[0] = ('pygame.%s.%s?'%(modname,name))+docs[0]
                        documents.append([title] + docs)
                except AttributeError: pass

        elif type(obj) in (types.FunctionType,):# types.BuiltinFunctionType):
            if not hasattr(obj, '__doc__'): continue
            title = '    /*DOC*/ static char doc_' + name + '[] =\n'
            doclines = getpydoclines(obj.__doc__)
            if doclines:
                documents.append([title] + doclines)
    return documents


def parsedocs(docs):
    modules = {}
    extras = {}
    funcs = []

    for d in docs:
        #print d
        modpos = d[0].find('_MODULE')
        extpos = d[0].find('_EXTRA')

        if modpos != -1:
            start = d[0].rfind(' ', 0, modpos)
            name = d[0][start+5:modpos]
            modules[name] = '\n'.join(d[1:])
        elif extpos != -1:
            start = d[0].rfind(' ', 0, extpos)
            name = d[0][start+5:extpos]
            extras[name] = '\n'.join(d[1:])
        else:
            obj = {'docs':['no documentation']}
            name = d[1][:d[1].find('(')]
            dot = name.rfind('.')
            name = name.replace('?', '.')
            if dot == -1:
                print 'NODOTFOUND, MISC:', name, dot
                obj['category'] = 'misc'
                obj['name'] = name
                obj['fullname'] = name
            else:
                obj['category'] = name[:dot].replace('.', '_')
                obj['name'] = name[dot+1:]
                obj['fullname'] = name
            try:
                obj['usage'] = d[1].replace('?',  '.')
                obj['quick'] = d[2]
                obj['docs'] = d[4:]
            except IndexError: pass
            #print '##APPEND FUNC:', obj['name']
            funcs.append(obj)

    return [modules, extras, funcs]


def getdocinfo(file, prefix=''):
    path = '/'.join((prefix, file[3:]))[1:]
    prettyname = (os.path.splitext(os.path.split(file)[-1])[0])
    prettyname = prettyname[0].capitalize() + prettyname[1:]
    if file.find('html') >= 0:
        info = open(file).readlines(2)[1]
    else:
        info = open(file).readline().strip().capitalize()
    return path, prettyname, info


def findtutorials():
    fileline = '<li><a href=%s>%s</a> - %s</li>'
    texthead = '<font size=+1><b>Included With Pygame</b></font><br>'
    tuthead = '<font size=+2><b>Tutorials</b></font><br>'
    texts1 =  ['../../readme.html', '../../install.html', '../LGPL', '../logos.html']
    texts = [fileline%x for x in [getdocinfo(x) for x in texts1]]
    finaltext = texthead + '\n'.join(texts)
    htmlfiles = glob.glob('../tut/*.html') + glob.glob('../tut/*/*.html')
    tuts1 =  [x.replace('\\','/') for x in htmlfiles]
    tuts1.sort()
    tuts = [fileline%(x[0],x[1],x[2][9:]) for x in [getdocinfo(x) for x in tuts1] if x[2].startswith('TUTORIAL:')]
    finaltut = tuthead + '\n'.join(tuts)
    return finaltut + '<br>&nbsp;<br>' + finaltext


def lookupdoc(docs, name, category):
    for d in docs:
        if d['fullname'] == category + '.' + name:
            return d
        if d['fullname'] == name or d['name'] == name:
            return d


def htmlize(doc, obj, func):
    for i in range(len(doc)):
        line = doc[i]
        line = line.replace('<<', '&lt;&lt;').replace('>>', '&gt;&gt;')
        pos = 0
        while 1:
            pos = line.find('(', pos+1)
            if pos == -1: break
            if line[pos-1].isspace(): continue
            start = line.rfind(' ', 0, pos)
            start2 = line.rfind('\n', 0, pos)
            start = max(max(start, start2), 0)
            lookname = line[start+1:pos]
            if lookname.startswith('ygame'):
                lookname = 'p' + lookname
                start -= 1
            elif lookname[1:].startswith('pygame'):
                lookname = lookname[1:]
                start += 1
            match = lookupdoc(func, lookname, obj['category'])
            if not match:
                #print 'NOMATCH: "'+ obj['category'] +'" "' + lookname + '"'
                continue
            end = line.find(')', pos)+1
            if match['fullname'] == obj['fullname']:
                link = '<u>%s</u>' % line[start+1:end]
            else:
                if match['category'] == obj['category']:
                    dest = '#%s' % (match['name'])
                else:
                    dest = '%s.html#%s' % (match['category'], match['name'])
                link = '<a href=%s>%s</a>' % (dest, line[start+1:end])
            line = line[:start+1] + link + line[end:]
            pos += len(link) - (pos-start)
        doc[i] = line


def buildlinks(alldocs):
    mod, ext, func = alldocs
    for obj in func:
        doc = obj['docs']
        htmlize(doc, obj, func)
    for k,i in mod.items():
        doc = [i]
        obj = {'category': k, 'fullname':''}
        htmlize(doc, obj, func)
        mod[k] = doc[0]
    for k,i in ext.items():
        doc = [i]
        obj = {'category': k, 'fullname':''}
        htmlize(doc, obj, func)
        ext[k] = doc[0]


def categorize(allfuncs):
    cats = {}
    for f in allfuncs:
        cat = f['category']
        if not cats.has_key(cat):
            cats[cat] = []
        cats[cat].append(f)
    return cats



def create_toc(allfuncs, prefix=''):
    mods = {}
    for f in allfuncs:
        cat = f['category']
        mods[cat] = ''

    l_mod = []
    l_py = []
    l_type = []
    for m in mods.keys():
        if m[7:] in PYTHONSRC:
            l = l_py
        elif m[0].lower() == m[0]:
            l = l_mod
        else:
            l = l_type

        file = m.replace('.', '_') + '.html'
        if m[:7] == 'pygame_':
            m = m[7:]
        str = '<a href=%s%s>%s</a>' % (prefix, file, m)
        l.append(str)

    l_py.sort()
    l_mod.sort()
    l_type.sort()

    str = ''
    items_per_line = 6
    for x in range(0, len(l_mod), items_per_line):
        row = l_mod[x:x+items_per_line]
        str += '|| ' + ' || \n'.join(row) + ' ||<br>\n'
    str += '&nbsp;<br>'
    for x in range(0, len(l_type), items_per_line):
        row = l_type[x:x+items_per_line]
        str += '|| ' + ' || \n'.join(row) + ' ||<br>\n'
    str += '&nbsp;<br>'
    for x in range(0, len(l_py), items_per_line):
        row = l_py[x:x+items_per_line]
        str += '|| ' + ' || \n'.join(row) + ' ||<br>\n'

    return str


def namesort(a,b): return cmp(a['name'], b['name'])


def writefuncdoc(alldocs):
    modules, extras, funcs = alldocs
    for cat, docs in funcs.items():
        htmldocs = []
        htmllist = []
        docs.sort(namesort)
        for d in docs:
            d['docs'] = '\n'.join(d['docs'])
            htmldocs.append(filltemplate(DOCTEMPLATE, d))
            htmllist.append(filltemplate(LISTTEMPLATE, d))
        modinfo = modules.get(cat, '')
        extrainfo = extras.get(cat, None)
        if extrainfo:
            modinfo += '<p>&nbsp;</p>' + extrainfo

        finalinfo = {'title': cat.replace('_', '.'),
                     'docs': '\n'.join(htmldocs),
                     'index': '\n'.join(htmllist),
                     'toc': MODULETOC,
                     'module': modinfo,
                     'mainpage': '../index.html',
                     'logo': '../pygame_tiny.gif'}
        page = filltemplate(PAGETEMPLATE, finalinfo)
        file = open(OUTPUTDIR + cat + '.html', 'w')
        file.write(page)
        file.close()



def makefullindex(alldocs):
    modules, extras, funcs = alldocs
    fullindex = []
    for cat, docs in funcs.items():
        htmldocs = []
        htmllist = []
        docs.sort(namesort)
        for d in docs:
            name = d['category'].replace('_', '.')
            if name.startswith('pygame.'):
                name = name[7:]
            d['mod'] = name
            s = filltemplate(INDEXTEMPLATE, d)
            fullindex.append(s)
    fullindex.sort()
    return INDEXSTART + ''.join(fullindex) + INDEXEND



def main():
    #find all sources
    files = []
    for s in SOURCES:
        files += glob.glob(s)
    for f in files[:]:
        if os.path.split(f)[1] in IGNORE_SOURCES:
            files.remove(f)

    #load all sources
    print 'read c sources...'
    rawdocs = []
    for f in files:
        rawdocs += readsource(f)

    print 'read python sources...'
    for f in PYTHONSRC:
        rawdocs += readpysource(f)


    #parse sources
    alldocs = parsedocs(rawdocs)

    #find and create hyperlinks
    buildlinks(alldocs)

    #create table of contents
    global MODULETOC
    pathed_toc = create_toc(alldocs[2], 'ref/')
    MODULETOC = create_toc(alldocs[2])

    #categorize
    alldocs[2] = categorize(alldocs[2])
    #write html
    print 'writing...'
    writefuncdoc(alldocs)

    fulldocs = findtutorials() + makefullindex(alldocs)
    fulldocs = fulldocs



    #create index
    finalinfo = {'title': 'Pygame Documentation',
                 'docs': fulldocs,
                 'index': mainindex_desc,
                 'toc': pathed_toc,
                 'module': ' ',
                 'mainpage': 'index.html',
                 'logo': 'pygame_tiny.gif',
                }
    page = filltemplate(PAGETEMPLATE, finalinfo)
    file = open('../index.html', 'w')
    file.write(page)
    file.close()


if __name__ == '__main__':
    main()
