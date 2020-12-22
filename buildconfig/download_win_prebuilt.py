import os
import stat
import logging

download_dir = "prebuilt_downloads"

def download_sha1_unzip(url, checksum, save_to_directory, unzip=True):
    """ This
    - downloads a url,
    - sha1 checksum check,
    - save_to_directory,
    - then unzips it.

    Does not download again if the file is there.
    Does not unzip again if the file is there.
    """
    try:
        import urllib.request as urllib
    except ImportError:
        import urllib2 as urllib
    import hashlib
    import zipfile

    filename = os.path.split(url)[-1]
    save_to = os.path.join(save_to_directory, filename)

    # skip download?
    skip_download = os.path.exists(save_to)
    if skip_download:
        with open(save_to, 'rb') as the_file:
            data = the_file.read()
            cont_checksum = hashlib.sha1(data).hexdigest()
            if cont_checksum == checksum:
                print("Skipping download url:%s: save_to:%s:" % (url, save_to))
    else:
        print("Downloading...", url, checksum)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, '
                                 'like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
        request = urllib.Request(url, headers=headers)
        response = urllib.urlopen(request).read()
        cont_checksum = hashlib.sha1(response).hexdigest()
        if checksum != cont_checksum:
            raise ValueError(
                'url:%s should have checksum:%s: Has:%s: ' % (url, checksum, cont_checksum)
            )
        with open(save_to, 'wb') as f:
            f.write(response)

    if unzip and filename.endswith('.zip'):
        print("Unzipping :%s:" % save_to)
        with zipfile.ZipFile(save_to, 'r') as zip_ref:
            zip_dir = os.path.join(
                save_to_directory,
                filename.replace('.zip', '')
            )
            if os.path.exists(zip_dir):
                print("Skipping unzip to zip_dir exists:%s:" % zip_dir)
            else:
                os.mkdir(zip_dir)
                zip_ref.extractall(zip_dir)

def get_urls(x86=True, x64=True, sdl2=True):
    url_sha1 = []
    if sdl2:
        url_sha1.extend([
            [
            'https://www.libsdl.org/release/SDL2-devel-2.0.14-VC.zip',
            '48d5dcd4a445410301f5575219cffb6de654edb8',
            ],
            [
            'https://www.libsdl.org/projects/SDL_image/release/SDL2_image-devel-2.0.5-VC.zip',
            '137f86474691f4e12e76e07d58d5920c8d844d5b',
            ],
            [
            'https://www.libsdl.org/projects/SDL_ttf/release/SDL2_ttf-devel-2.0.15-VC.zip',
            '1436df41ebc47ac36e02ec9bda5699e80ff9bd27',
            ],
            [
            'https://www.libsdl.org/projects/SDL_mixer/release/SDL2_mixer-devel-2.0.4-VC.zip',
            '9097148f4529cf19f805ccd007618dec280f0ecc',
            ],
            [
            'https://www.ijg.org/files/jpegsr9d.zip',
            'ed10aa2b5a0fcfe74f8a6f7611aeb346b06a1f99',
            ],
        ])
    if x86:
        url_sha1.append([
         'https://pygame.org/ftp/prebuilt-x86-pygame-1.9.2-20150922.zip',
         'dbce1d5ea27b3da17273e047826d172e1c34b478'
        ])
    if x64:
        url_sha1.append([
         'https://pygame.org/ftp/prebuilt-x64-pygame-1.9.2-20150922.zip',
         '3a5af3427b3aa13a0aaf5c4cb08daaed341613ed'
        ])
    return url_sha1

def download_prebuilts(temp_dir, x86=True, x64=True, sdl2=True):
    """ For downloading prebuilt dependencies.
    """
    from distutils.dir_util import mkpath
    if not os.path.exists(temp_dir):
        print("Making dir :%s:" % temp_dir)
        mkpath(temp_dir)
    for url, checksum in get_urls(x86=x86, x64=x64, sdl2=sdl2):
        download_sha1_unzip(url, checksum, temp_dir, 1)

def create_ignore_target_fnc(x64=False, x86=False):
    if not x64 and not x86:
        return None
    strs = []
    if x64:
        strs.append('x64')
    if x86:
        strs.append('x86')
    def ignore_func(dir, contents):
        for target in strs:
            if target in dir:
                return contents
        return []
    return ignore_func

import shutil

def copytree(src, dst, symlinks=False, ignore=None):
    """like shutil.copytree() but ignores existing files
    https://stackoverflow.com/a/22331852/1239986
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except OSError:
                pass # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def place_downloaded_prebuilts(temp_dir, move_to_dir, x86=True, x64=True, sdl2=True):
    """ puts the downloaded prebuilt files into the right place.

    Leaves the files in temp_dir. copies to move_to_dir
    """
    prebuilt_x64 = os.path.join(
        temp_dir,
        'prebuilt-x64-pygame-1.9.2-20150922',
        'prebuilt-x64'
    )
    prebuilt_x86 = os.path.join(
        temp_dir,
        'prebuilt-x86-pygame-1.9.2-20150922',
        'prebuilt-x86'
    )

    ignore = None
    def copy(src, dst):
        copytree(src, dst, ignore=ignore)

    if x64:
        copy(prebuilt_x64, os.path.join(move_to_dir, 'prebuilt-x64'))
    if x86:
        copy(prebuilt_x86, os.path.join(move_to_dir, 'prebuilt-x86'))

    ignore = create_ignore_target_fnc(x64=not x64, x86=not x86)
    prebuilt_dirs = []
    if x86:
        prebuilt_dirs.append('prebuilt-x86')
    if x64:
        prebuilt_dirs.append('prebuilt-x64')

    if not sdl2:
        return

    for prebuilt_dir in prebuilt_dirs:
        path = os.path.join(move_to_dir, prebuilt_dir)
        print("copying into %s" % path)

        # update jpeg
        for file in ('jerror.h', 'jmorecfg.h', 'jpeglib.h'):
            shutil.copyfile(
                os.path.join(
                    temp_dir,
                    'jpegsr9d',
                    'jpeg-9d',
                    file
                ),
                os.path.join(
                    move_to_dir,
                    prebuilt_dir,
                    'include',
                    file
                )
            )

        copy(
            os.path.join(
                temp_dir,
                'SDL2_image-devel-2.0.5-VC/SDL2_image-2.0.5'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_image-2.0.5'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2_mixer-devel-2.0.4-VC/SDL2_mixer-2.0.4'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_mixer-2.0.4'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2_ttf-devel-2.0.15-VC/SDL2_ttf-2.0.15'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_ttf-2.0.15'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2-devel-2.0.14-VC/SDL2-2.0.14'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2-2.0.14'
            )
        )

def update(x86=True, x64=True, sdl2=True):
    move_to_dir = "."
    download_prebuilts(download_dir, x86=x86, x64=x64, sdl2=sdl2)
    place_downloaded_prebuilts(download_dir, move_to_dir, x86=x86, x64=x64, sdl2=sdl2)

def ask(x86=True, x64=True, sdl2=True):
    move_to_dir = "."
    if x64:
        dest_str = "\"%s/prebuilt-x64\"" % move_to_dir
    else:
        dest_str = ""
    if x86:
        if dest_str:
            dest_str = "%s and " % dest_str
        dest_str = "%s\"%s/prebuilt-x86\"" % (dest_str, move_to_dir)
    logging.info('Downloading prebuilts to "%s" and copying to %s.', (download_dir, dest_str))
    download_prebuilt = True

    if download_prebuilt:
        update(x86=x86, x64=x64, sdl2=sdl2)
    return download_prebuilt

def cached(x86=True, x64=True, sdl2=True):
    if not os.path.isdir(download_dir):
        return False
    for url, check in get_urls(x86=x86, x64=x64, sdl2=sdl2):
        filename = os.path.split(url)[-1]
        save_to = os.path.join(download_dir, filename)
        if not os.path.exists(save_to):
            return False
    return True

if __name__ == '__main__':
    ask()
