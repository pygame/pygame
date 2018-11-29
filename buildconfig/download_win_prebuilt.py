import os

try:
    raw_input
except NameError:
    raw_input = input


def download_sha1_unzip(url, checksum, save_to_directory, unzip=True):
    """ This
    - downloads a url,
    - sha1 checksum check,
    - save_to_directory,
    - then unzips it.

    Does not download again if the file is there.
    Does not unzip again if the file is there.
    """
    import requests
    import hashlib
    import zipfile

    filename = os.path.split(url)[-1]
    save_to = os.path.join(save_to_directory, filename)

    download_file = True
    # skip download?
    skip_download = os.path.exists(save_to)
    if skip_download:
        with open(save_to, 'rb') as the_file:
            data = the_file.read()
            cont_checksum = hashlib.sha1(data).hexdigest()
            if cont_checksum == checksum:
                download_file = False
                print("Skipping download url:%s: save_to:%s:" % (url, save_to))
    else:
        print("Downloading...", url, checksum)
        response = requests.get(url)
        cont_checksum = hashlib.sha1(response.content).hexdigest()
        if checksum != cont_checksum:
            raise ValueError(
                'url:%s should have checksum:%s: Has:%s: ' % (url, checksum, cont_checksum)
            )
        with open(save_to, 'wb') as f:
            f.write(response.content)

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


def download_prebuilts(temp_dir):
    """ For downloading prebuilt dependencies.
    """
    from distutils.dir_util import mkpath
    if not os.path.exists(temp_dir):
        print("Making dir :%s:" % temp_dir)
        mkpath(temp_dir)
    url_sha1 = [
        [
        'https://www.libsdl.org/release/SDL2-devel-2.0.9-VC.zip',
        '0b4d2a9bd0c66847d669ae664c5b9e2ae5cc8f00',
        ],
        [
        'https://www.libsdl.org/projects/SDL_image/release/SDL2_image-devel-2.0.4-VC.zip',
        'f5199c52b3af2e059ec0268d4fe1854311045959',
        ],
        [
        'https://www.libsdl.org/projects/SDL_ttf/release/SDL2_ttf-devel-2.0.14-VC.zip',
        'c64d90c1f7d1bb3f3dcfcc255074611f017cdcc4',
        ],
        [
        'https://www.libsdl.org/projects/SDL_mixer/release/SDL2_mixer-devel-2.0.4-VC.zip',
        '9097148f4529cf19f805ccd007618dec280f0ecc',
        ],
        # [
        #  'https://www.libsdl.org/release/SDL2-2.0.9-win32-x86.zip',
        #  '04a48d0b429ac65f0d9b33bd1b75d77526c0cccf'
        # ],
        # [
        #  'https://www.libsdl.org/release/SDL2-2.0.9-win32-x64.zip',
        #  '7a156a8c81d2442901dea90ff0f71026475e89c6'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_ttf/release/SDL2_ttf-2.0.14-win32-x86.zip',
        #  '0c89aa4097745ac68516783b7fd67abd019b7701'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_ttf/release/SDL2_ttf-2.0.14-win32-x64.zip',
        #  '47446c907d006804e12ecd827a45dcc89abd2264'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_image/release/SDL2_image-2.0.4-win32-x86.zip',
        #  'e9b8b84edfe618bec73f91111324e37c37dd6f27'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_image/release/SDL2_image-2.0.4-win32-x64.zip',
        #  '956750cb442264abd8cd398c57aa493249cf04d4'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_mixer/release/SDL2_mixer-2.0.4-win32-x86.zip',
        #  '0bfc276a3d50613ae54831ff196721ad24de1432'
        # ],
        # [
        #  'https://www.libsdl.org/projects/SDL_mixer/release/SDL2_mixer-2.0.4-win32-x64.zip',
        #  'afa34e9c11fd8a6f5d084862c38fcf0abdc77514'
        # ],
        [
         'https://bitbucket.org/llindstrom/pygame/downloads/prebuilt-x86-pygame-1.9.2-20150922.zip',
         'dbce1d5ea27b3da17273e047826d172e1c34b478'
        ],
        [
         'https://bitbucket.org/llindstrom/pygame/downloads/prebuilt-x64-pygame-1.9.2-20150922.zip',
         '3a5af3427b3aa13a0aaf5c4cb08daaed341613ed'
        ],
    ]
    for url, checksum in url_sha1:
        download_sha1_unzip(url, checksum, temp_dir, 1)

def place_downloaded_prebuilts(temp_dir, move_to_dir):
    """ puts the downloaded prebuilt files into the right place.

    Leaves the files in temp_dir. copies to move_to_dir
    """
    import shutil
    import distutils.dir_util
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

    def copy(src, dst):
        return distutils.dir_util.copy_tree(
            src, dst, preserve_mode=1, preserve_times=1
        )

    copy(prebuilt_x64, os.path.join(move_to_dir, 'prebuilt-x64'))
    copy(prebuilt_x86, os.path.join(move_to_dir, 'prebuilt-x86'))

    # For now...
    # copy them into both folders. Even though they contain different ones.
    for prebuilt_dir in ['prebuilt-x64', 'prebuilt-x86']:
        path = os.path.join(move_to_dir, prebuilt_dir)
        print("copying into %s" % path)
        copy(
            os.path.join(
                temp_dir,
                'SDL2_image-devel-2.0.4-VC/SDL2_image-2.0.4'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_image-2.0.4'
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
                'SDL2_ttf-devel-2.0.14-VC/SDL2_ttf-2.0.14'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2_ttf-2.0.14'
            )
        )
        copy(
            os.path.join(
                temp_dir,
                'SDL2-devel-2.0.9-VC/SDL2-2.0.9'
            ),
            os.path.join(
                move_to_dir,
                prebuilt_dir,
                'SDL2-2.0.9'
            )
        )

def ask():
    temp_dir = "prebuilt_downloads"
    move_to_dir = "."
    reply = raw_input(
            '\nDownload prebuilts to "%s" and copy to "%s/prebuilt-x64" and "%s/prebuilt-x86"? [Y/n]' % (temp_dir, move_to_dir, move_to_dir))
    download_prebuilt = (not reply) or reply[0].lower() != 'n'

    if download_prebuilt:
        download_prebuilts(temp_dir)
        place_downloaded_prebuilts(temp_dir, move_to_dir)

if __name__ == '__main__':
    ask()
