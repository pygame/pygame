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
    # requests does connection retrying, but people might not have it installed.
    use_requests = True

    try:
        import requests
    except ImportError:
        use_requests = False

    import urllib.request as urllib
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
                print(f"Skipping download url:{url}: save_to:{save_to}:")
    else:
        print("Downloading...", url, checksum)

        if use_requests:
            response = requests.get(url)
            cont_checksum = hashlib.sha1(response.content).hexdigest()
        else:

            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, '
                                     'like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
            request = urllib.Request(url, headers=headers)
            response = urllib.urlopen(request).read()
            cont_checksum = hashlib.sha1(response).hexdigest()

        if checksum != cont_checksum:
            raise ValueError(
                f'url:{url} should have checksum:{checksum}: Has:{cont_checksum}: '
            )
        with open(save_to, 'wb') as f:
            if use_requests:
                f.write(response.content)
            else:
                f.write(response)

    if unzip and filename.endswith('.zip'):
        print(f"Unzipping :{save_to}:")
        with zipfile.ZipFile(save_to, 'r') as zip_ref:
            zip_dir = os.path.join(
                save_to_directory,
                filename.replace('.zip', '')
            )
            if os.path.exists(zip_dir):
                print(f"Skipping unzip to zip_dir exists:{zip_dir}:")
            else:
                os.mkdir(zip_dir)
                zip_ref.extractall(zip_dir)

def get_urls(x86=True, x64=True, ARM64=True):
    url_sha1 = []
    url_sha1.extend([
        [
        'https://github.com/RockLakeGrass/SDL-Windows-ARM64/releases/download/SDL2-pygame-deps/SDL2-devel-2.0.22-VC.zip',
        '4606b4852ef71f316d96c9138d94eee27993bcb4',
        ],
        [
        'https://github.com/RockLakeGrass/SDL-Windows-ARM64/releases/download/SDL2-pygame-deps/SDL2_image-devel-2.0.5-VC.zip',
        '8bfcbd64d8711e6cb07faf8bb2c3f452a1416422',
        ],
        [
        'https://github.com/RockLakeGrass/SDL-Windows-ARM64/releases/download/SDL2-pygame-deps/SDL2_ttf-devel-2.20.0-VC.zip',
        '33a8315a726c452855bf213df81e7ebeb46e246b'
        ],
        [
        'https://github.com/RockLakeGrass/SDL-Windows-ARM64/releases/download/SDL2-pygame-deps/SDL2_mixer-devel-2.6.0-VC.zip',
        '46df71b401cfb2a63a213a195b192118feaa98ad',
        ],
    ])
    if x86:
        url_sha1.append([
         'https://github.com/pygame/pygame/releases/download/2.1.3.dev4/prebuilt-x86-pygame-2.1.4-20220319.zip',
         'bff2e50d65ec35274d33203e9fcaf5d53b31a696'
        ])
    if x64:
        url_sha1.append([
         'https://github.com/pygame/pygame/releases/download/2.1.3.dev4/prebuilt-x64-pygame-2.1.4-20220319.zip',
         '16b46596744ce9ef80e7e40fa72ddbafef1cf586'
        ])
    if ARM64:
        url_sha1.append([
         'https://github.com/RockLakeGrass/pygame/releases/download/2.1.3.dev5/prebuilt-ARM64-pygame.zip',
         'fd0a1645a05f3496d6195edcf635133c10937660'
        ])
    return url_sha1

def download_prebuilts(temp_dir, x86=True, x64=True, ARM64=True):
    """ For downloading prebuilt dependencies.
    """
    if not os.path.exists(temp_dir):
        print(f"Making dir :{temp_dir}:")
        os.makedirs(temp_dir)
    for url, checksum in get_urls(x86=x86, x64=x64, ARM64=ARM64):
        download_sha1_unzip(url, checksum, temp_dir, 1)

def create_ignore_target_fnc(x64=False, x86=False, ARM64 = False):
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

def place_downloaded_prebuilts(temp_dir, move_to_dir, x86=True, x64=True, ARM64=True):
    """ puts the downloaded prebuilt files into the right place.

    Leaves the files in temp_dir. copies to move_to_dir
    """
    prebuilt_x64 = os.path.join(
        temp_dir,
        'prebuilt-x64-pygame-2.1.4-20220319',
        'prebuilt-x64'
    )
    prebuilt_x86 = os.path.join(
        temp_dir,
        'prebuilt-x86-pygame-2.1.4-20220319',
        'prebuilt-x86'
    )
    prebuilt_ARM64 = os.path.join(
        temp_dir,
        'prebuilt-ARM64-pygame',
        'prebuilt-ARM64'
    )

    ignore = None
    def copy(src, dst):
        copytree(src, dst, ignore=ignore)

    if x64:
        copy(prebuilt_x64, os.path.join(move_to_dir, 'prebuilt-x64'))
    if x86:
        copy(prebuilt_x86, os.path.join(move_to_dir, 'prebuilt-x86'))
    if ARM64:
        copy(prebuilt_ARM64, os.path.join(move_to_dir, 'prebuilt-ARM64'))

    ignore = create_ignore_target_fnc(x64=not x64, x86=not x86, ARM64=not ARM64)
    prebuilt_dirs = []
    if x86:
        prebuilt_dirs.append('prebuilt-x86')
    if x64:
        prebuilt_dirs.append('prebuilt-x64')
    if ARM64:
        prebuilt_dirs.append('prebuilt-ARM64')

    for prebuilt_dir in prebuilt_dirs:
        path = os.path.join(move_to_dir, prebuilt_dir)
        print(f"copying into {path}")

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
                'SDL2_mixer-devel-2.6.0-VC/SDL2_mixer-2.6.0'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_mixer-2.6.0'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2_ttf-devel-2.20.0-VC/SDL2_ttf-2.20.0'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_ttf-2.20.0'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2-devel-2.0.22-VC/SDL2-2.0.22'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2-2.0.22'
            )
        )

def update(x86=True, x64=True, ARM64=True):
    move_to_dir = "."
    download_prebuilts(download_dir, x86=x86, x64=x64, ARM64=ARM64)
    place_downloaded_prebuilts(download_dir, move_to_dir, x86=x86, x64=x64, ARM64=ARM64)

def ask(x86=True, x64=True, ARM64=True):
    move_to_dir = "."
    if x64:
        dest_str = f"\"{move_to_dir}/prebuilt-x64\""
    elif ARM64:
        dest_str = f"\"{move_to_dir}/prebuilt-ARM64\""
    else:
        dest_str = ""
    if x86:
        if dest_str:
            dest_str = f"{dest_str} and "
        dest_str = f"{dest_str}\"{move_to_dir}/prebuilt-x86\""
    logging.info('Downloading prebuilts to "%s" and copying to %s.', (download_dir, dest_str))
    download_prebuilt = True

    if download_prebuilt:
        update(x86=x86, x64=x64, ARM64=ARM64)
    return download_prebuilt

def cached(x86=True, x64=True, ARM64=True):
    if not os.path.isdir(download_dir):
        return False
    for url, check in get_urls(x86=x86, x64=x64, ARM64=ARM64):
        filename = os.path.split(url)[-1]
        save_to = os.path.join(download_dir, filename)
        if not os.path.exists(save_to):
            return False
    return True

if __name__ == '__main__':
    ask()
