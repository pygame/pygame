
"""take a C file, extract the docstring, and turn it into
a simple textfile for easy editing"""

import fnmatch

docpattern = '*static char*doc_*=*'

def src2txt(infile, outfile):
    global commentpattern
    if type(infile) == type('x'):
        infile = open(infile)
    if type(outfile) == type('x'):
        outfile = open(outfile, 'w')
    while 1:
        line = infile.readline()
        if not line: break
        if fnmatch.fnmatch(line, docpattern) and line.find(';') == -1:
            docname = line[:line.find('[')]
            docname = docname[docname.rfind(' '):]
            outfile.write('#START:'+ docname+ '\n')
            docs = []
            while 1:
                line = infile.readline()
                if not line: break
                if line.find('"') == -1:
                    if line.find(';') != -1:
                        break
                    continue
                origline = line
                line = line[line.find('"'):]
                line = line.replace(';', ' ')                
                line = line.replace('"', ' ').strip()
                if line[-2:] == '\\n':
                    line = line[:-2]
                if not line:
                    line = '\n\n'
                else:
                    line += ' '
                #outfile.write(line)
                docs.append(line)
                if origline.find(';') != -1: break;
            if docname.find('MODULE') == -1 and docname.find('EXTRA') == -1:
                docs[0] += '\n'
                docs[1] += '\n'
                docs[2] = '\n'
            outfile.write(''.join(docs))
            outfile.write('\n#END\n\n');

                




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
