
"""take a C file, extract the docstring, and turn it into
a simple textfile for easy editing"""

import fnmatch

docpattern = '#Start:*'
docprefix = '    /*DOC*/ '


def wordwrap(line, width):
    "split text into a list of wordwrapped text"
    words = line.split()
    outlines = []
    curline = ''
    for w in words:
        if len(curline) + len(w) > width:
            outlines.append(curline[1:])
            curline = ''
        curline += ' '+w
    outlines.append(curline[1:])
    return outlines



def txt2src(infile, outfile):
    global commentpattern
    if type(infile) == type('x'):
        infile = open(infile)
    if type(outfile) == type('x'):
        outfile = open(outfile, 'w')
    while 1:
        line = infile.readline()
        if not line: break
        if fnmatch.fnmatch(line, docpattern):
            docname = line.split()[1]
            outfile.write(docprefix + 'static char %s[] =\n'%docname)
            while 1:
                line = infile.readline()
                line = line.strip()
                if line == '#END': break
                outlines = wordwrap(line, 50)
                for l in outlines:
                    outfile.write(docprefix + '   "%s\\n"\n'%l)
            outfile.write(docprefix + ';\n\n')
                




if __name__ == '__main__':
    import sys
    try:
        inname = sys.argv[1]
        outname = sys.argv[2]
    except IndexError:
        raise SystemExit, 'USAGE: src2txt infile outfile'
    try:
        infile = open(inname, 'r')
        outfile = open(outname, 'w')
    except IOError:
        raise SystemExit, 'Error: Cannot open files'
    txt2src(infile, outfile)
