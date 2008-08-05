################################################################################

import re
import os
import zipfile

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
            
def prepare_dir(dir):
    if os.path.exists(dir): cleardir(dir)
    else: os.mkdir(dir)

def create_zip_from_dict(zip_file, files):
    zip = zipfile.ZipFile(zip_file, 'w')
    for k, v in files.items():
        zip.writestr(k, v)
    zip.close()

################################################################################

if __name__ == '__main__':
    create_zip_from_dict('test.zip', {'one.txt':'hello world'})
    assert os.path.exists('test.zip')
    os.remove('test.zip')
    print 'OK'