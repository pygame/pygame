#!/usr/bin/env python

import os, glob


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

def Run():
    
    from optparse import OptionParser    
    parser = OptionParser()
    parser.add_option("", "--no-code-docs", dest="have_code_docs",
                      action="store_false", default=True,
                      help="No python documentation in code.")
    (options, args) = parser.parse_args()

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
        fullname = os.path.join("docs","ref","%s.html") % name
        outFile = open(fullname, "w")
        outFile.write(HTMLHeader % name)
        WritePageLinks(outFile, pages)
        outFile.write(HTMLMid)
        HtmlOut(doc, index, outFile)
        outFile.write(HTMLFinish)
        outFile.close()
 
    outFile = open(os.path.join("src", "pygamedocs.h"), "w")
    outFile.write("/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */\n")
    for doc in justDocs:
        WriteDocHeader(outFile, doc, options.have_code_docs)


    outFile.write("\n\n/* Docs in a comments... slightly easier to read. */\n\n\n/*")
    # add the docs as comments to the header file.
    for doc in justDocs:
        WriteDocHeaderComments(outFile, doc)

    outFile.write("\n\n*/\n\n")


    topDoc = LayoutDocs(justDocs)

    outFile = open(os.path.join("docs","ref","index.html"), "w")
    outFile.write(HTMLHeader % "Index")
    WritePageLinks(outFile, pages)
    outFile.write(HTMLMid)
    outFile.write("<ul>\n\n")
    WriteIndex(outFile, index, topDoc)
    outFile.write("\n\n</ul>\n")
    outFile.write(HTMLFinish)
    outFile.close()


def HtmlOut(doc, index, f):
    f.write('\n\n<a name="%s">\n' % doc.fullname)
    f.write("<big><b>%s</big></b><br><ul>\n" % doc.fullname)
    if doc.descr:
        f.write("  <i>%s</i><br>\n" % doc.descr) 
    if doc.protos:
        for p in doc.protos:
            f.write("  <tt>%s</tt><br>\n" % p)
    if doc.kids:
        f.write("<ul><small><table>\n")
        for kid in doc.kids:
            f.write("  <tr><td>%s</td><td>%s</td></tr>\n"
                        % (index.get(kid.fullname + "()"), kid.descr or ""))
        f.write("</table></small></ul>\n")
    if doc.docs:
        pre = False
        for d in doc.docs:
            if d[0] == '*':
                f.write("<ul>\n")
                for li in d[1:].split('*'):
                    txt = HtmlPrettyWord(li)
                    f.write(" <li>%s</li>\n" % txt)
                f.write("</ul>\n")
            else:
                txt, pre = HtmlPrettyLine(d, index, pre)
                f.write(txt)
        if pre:
            f.write("</pre>\n")
    else:
        f.write(" &nbsp;<br> \n")

    f.write("<!--COMMENTS:"+doc.fullname+"-->")
    f.write(" &nbsp;<br> \n")
    
    if doc.kids:
        for k in doc.kids:
            HtmlOut(k, index, f)
    f.write("<br></ul>\n")



def HtmlPrettyWord(word):
    if "." in word[:-1] or word.isupper():
        return "<tt>%s</tt>" % word
    return word



def HtmlPrettyLine(line, index, pre):
    pretty = ""
    
    if line[0].isspace():
        if not pre:
            pretty += "<pre>"
            pre = True
    elif pre:
        pre = False
        pretty += "</pre>"
    
    if not pre:
        pretty += "<p>"
        for word in line.split():
            if word[-1] in ",.":
                finish = word[-1]
                word = word[:-1]
            else:
                finish = ""
            link = index.get(word)
            if link:
                pretty += "<tt>%s</tt>%s " % (link, finish)
            elif word.isupper() or "." in word[1:-1]:
                pretty += "<tt>%s</tt>%s " % (word, finish)
            else:
                pretty += "%s%s " % (word, finish)
        pretty += "</p>\n"
    else:
        pretty += line + "\n"
    return pretty, pre



def WritePageLinks(outFile, pages):
    links = []
    for page in pages[1:]:
        link = '<a href="%s.html">%s</a>' % (page, page.title())
        links.append(link)
    outFile.write("&nbsp;||&nbsp;\n".join(links))
    #outFile.write("\n</p>\n\n")


def MakeIndex(name, doc, index):
    if doc.fullname:
        link = '<a href="%s.html#%s">%s</a> - <font size=-1>%s</font>' % (name, doc.fullname, doc.fullname, doc.descr)
        index[doc.fullname + "()"] = link
    if doc.kids:
        for kid in doc.kids:
            MakeIndex(name, kid, index)


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


def WriteIndex(outFile, index, doc):
    link = index.get(doc.fullname + "()", doc.fullname)
    outFile.write("<li>%s</li>\n" % link)
    if doc.kids:
        outFile.write("<ul>\n")
        sortKids = list(doc.kids)
        #print(sortKids)
        sortKids = sort_list_by_keyfunc(sortKids, lambda x: x.fullname)
        #sortKids = sorted( sortKids )
        for kid in sortKids:
            WriteIndex(outFile, index, kid)
        outFile.write("</ul>\n")



def WriteDocHeader(f, doc, have_code_docs ):
    name = doc.fullname.replace(".", "")
    name = name.replace("_", "")
    name = name.upper()
    defineName = "DOC_" + name
    text = ""
    if have_code_docs:
        if doc.protos:
            text = "\\n".join(doc.protos)
        if doc.descr:
            if text:
                text += "\\n"
            text += doc.descr
    
    f.write('#define %s "%s"\n\n' % (defineName, text))

    if doc.kids:
        for kid in doc.kids:
            WriteDocHeader(f, kid, have_code_docs)

def WriteDocHeaderComments(f, doc):
    name = doc.fullname

    defineName = name
    text = ""
    if doc.protos:
        text = "\n".join(doc.protos)
    if doc.descr:
        if text:
            text += "\n"
        text += doc.descr
    text = text.replace("\\n", "\n")
    #f.write('\n\n/*\n%s\n %s\n\n*/' % (defineName, text))
    f.write('\n\n%s\n %s\n\n' % (defineName, text))

    if doc.kids:
        for kid in doc.kids:
            WriteDocHeaderComments(f, kid)






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

HTMLHeader = """
<html>
<title>%s - Pygame Documentation</title>
<body bgcolor=#aaeebb text=#000000 link=#331111 vlink=#331111>


<table cellpadding=0 cellspacing=0 border=0 style='border: 3px solid black;' width='100%%'>
<tr>
<td bgcolor='#c2fc20' style='padding: 6px;' align=center valign=center><a href='http://www.pygame.org/'><img src='../pygame_tiny.gif' border=0 width=200 height=60></a><br><b>pygame documentation</b></td>
<td bgcolor='#6aee28' style='border-left: 3px solid black; padding: 6px;' align=center valign=center>
	||&nbsp;
	<a href=http://www.pygame.org>Pygame Home</a> &nbsp;||&nbsp;
	<a href=../index.html>Help Contents</a> &nbsp;||
	<a href=index.html>Reference Index</a> &nbsp;||
	<br>&nbsp;<br>
	
"""

HTMLMid = """
</td></tr></table>
<br>
"""

HTMLFinish = """
</body></html>
"""

if __name__ == '__main__':
    Run()
