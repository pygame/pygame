
"""take a C file, extract the docstring, and turn it into
a simple textfile for easy editing"""

import fnmatch

docpattern = 'static char*doc*=*'

def src2txt(infile, outfile):
    global commentpattern
    while 1:
        line = infile.readline()
        if not line: break
        if fnmatch.fnmatch(line, docpattern) and line.find(';') == -1:
            docname = line[:line.find('[')]
            docname = docname[docname.rfind(' '):]
            outfile.write('#START:'+ docname+ '\n')
            while 1:
                line = infile.readline()
                if not line or line.find(';') != -1: break
                line = line.replace('"', ' ').strip()
                if line[-2:] == '\\n':
                    line = line[:-2]
                outfile.write(line + '\n')
            outfile.write('#END\n\n');

                




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
    src2txt(infile, outfile)
