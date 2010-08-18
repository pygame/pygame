################################################################################

import re
import os
import sys
import time
import zipfile
import webbrowser

################################################################################

def write_file_lines(filename, line_list):
    file_obj = file(filename, "w")
    file_obj.writelines(['%s\n' % s for s in line_list])
    file_obj.close()
    
def re_sub_file(file_path, match, replace):
    content = file(file_path, "r").read()
    content, count = re.subn(match, replace, content)
    assert(count > 0)
    output = file(file_path, "w")
    output.write(content)
    output.close()

def assert_path_exists(path, description):
    if not os.path.exists(path):
        raise Exception("ERROR: can't find "+description+" at : "+path)

def cleardir(path_to_clear):
    for root, dirs, files in os.walk(path_to_clear, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def clear_or_make_dirs(dir):
    if os.path.exists(dir): cleardir(dir)
    else: os.makedirs(dir)

def create_zip(zip_file, *files, **var):
    zip = zipfile.ZipFile(zip_file, 'w', compression = zipfile.ZIP_DEFLATED)
    for f in files: zip.write(f, os.path.basename(f))
    for k, v in var.items(): zip.writestr(k, v)
    zip.close()

def dump_and_open_in_browser(string):
    write_file_lines('temp.html', [string])
    webbrowser.open('temp.html')                

def normp(*paths):
    return os.path.normpath(os.path.join(*paths))

################################################################################

def write_stdout(out): 
    sys.stdout.write(out)
    sys.stdout.flush()

class ProgressIndicator(object):
    def __init__(self, report_every = 1):
        self.progress = 1
        self.report_every = report_every

    def __call__(self):
        self.progress += 1
        if self.progress % self.report_every == 0:  write_stdout(".")

    def finish(self):
        write_stdout('\n')

################################################################################

if __name__ == '__main__':
    pass
    
################################################################################