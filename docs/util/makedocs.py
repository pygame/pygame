 
"""take all the PyGame source and create html documentation"""
#needs to eventually handle python source too.

import fnmatch, glob, os


DOCPATTERN = '*static char*doc_*=*'
SOURCES = ['../../src/*.c']
IGNORE_SOURCES = ['rwobject.c']
OUTPUTDIR = '../'

PAGETEMPLATE = open('pagelate.html').readlines()
DOCTEMPLATE = open('doclate.html').readlines()
LISTTEMPLATE = open('listlate.html').readlines()

MODULETOC = ""


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



def parsedocs(docs):
    modules = {}
    extras = {}
    funcs = []

    for d in docs:
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
            name = d[1][:d[1].rfind('(')]
            dot = name.rfind('.')
            if dot == -1:
                obj['category'] = 'misc'
                obj['name'] = name
                obj['fullname'] = name
            else:
                obj['category'] = name[:dot].replace('.', '_')
                obj['name'] = name[dot+1:]
                obj['fullname'] = name
            try:
                obj['usage'] = d[1]
                obj['quick'] = d[2]
                obj['docs'] = d[4:]
            except IndexError: pass
            funcs.append(obj)

    return [modules, extras, funcs]



def lookupdoc(docs, name, category):
    for d in docs:
        if d['fullname'] == category + '.' + name:
            return d
        if d['fullname'] == name or d['name'] == name:
            return d



def buildlinks(alldocs):
    mod, ext, func = alldocs
    for obj in func:
        doc = obj['docs']
        for i in range(len(doc)):
            line = doc[i]
            pos = 0
            while 1:
                pos = line.find('(', pos+1)
                if pos == -1: break
                if line[pos-1].isspace(): continue
                start = line.rfind(' ', 0, pos)
                if start == -1: continue
                lookname = line[start+1:pos]
                match = lookupdoc(func, lookname, obj['category'])
                if not match: continue
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



def categorize(allfuncs):
    cats = {}
    for f in allfuncs:
        cat = f['category']
        if not cats.has_key(cat):
            cats[cat] = []
        cats[cat].append(f)
    return cats



def create_toc(allfuncs):
    mods = {}
    for f in allfuncs:
        cat = f['category']
        mods[cat] = ''
    l = []
    for m in mods.keys():
        file = m.replace('.', '_') + '.html'
        if m[:7] == 'pygame_':
            m = m[7:]
        str = '<a href=%s>%s</a>' % (file, m)
        l.append(str)
    l.sort()
    str = ''
    for x in range(0, len(l), 5):
        row = l[x:x+5]
        str += '|| ' + ' || \n'.join(row) + ' ||<br>\n'
    global MODULETOC
    MODULETOC = str
    

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
            modinfo += '<br>&nbsp;<br>' + extrainfo

        finalinfo = {'title': cat.replace('_', '.'),
                     'docs': '\n'.join(htmldocs),
                     'index': '\n'.join(htmllist),
                     'toc': MODULETOC,
                     'module': modinfo}
        page = filltemplate(PAGETEMPLATE, finalinfo)
        file = open(OUTPUTDIR + cat + '.html', 'w')
        file.write(page)
        file.close()


def main():
    #find all sources
    files = []
    for s in SOURCES:
        files += glob.glob(s)
    for f in files[:]:
        if os.path.split(f)[1] in IGNORE_SOURCES:
            files.remove(f)

    #load all sources
    print 'read sources...'
    rawdocs = []
    for f in files:
        rawdocs += readsource(f)

    #parse sources
    alldocs = parsedocs(rawdocs)

    #find and create hyperlinks
    buildlinks(alldocs)

    #create table of contents
    create_toc(alldocs[2])

    #categorize
    alldocs[2] = categorize(alldocs[2])

    #write html
    print 'writing...'
    writefuncdoc(alldocs)


if __name__ == '__main__':
    main()