#!/usr/bin/env python

import sys
import os
import subprocess

# rst_dir = 'docs'
# rst_source_dir = os.path.join(rst_dir, 'reST')
# rst_build_dir = os.path.join('docs', 'generated')

# rst_source_dir = os.path.join(rst_dir, 'es')
# rst_build_dir = os.path.join('docs', 'generated', 'es')

# rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
# c_header_dir = os.path.join('src_c', 'doc')


def run():
    global rst_dir, rst_source_dir, rst_build_dir, rst_doctree_dir, c_header_dir
    rst_dir = 'docs'
    rst_source_dir = os.path.join(rst_dir, 'reST')
    rst_build_dir = os.path.join('docs', 'generated')

    rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
    c_header_dir = os.path.join('src_c', 'doc')
    print("Generating:", rst_source_dir, rst_build_dir)
    runit()


    rst_source_dir = os.path.join(rst_dir, 'es')
    rst_build_dir = os.path.join('docs', 'generated', 'es')
    rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
    print("Generating:", rst_source_dir, rst_build_dir)
    runit()


def runit():
    full_generation_flag = False
    for argument in sys.argv[1:]:
        if argument == 'full_generation':
            full_generation_flag = True
    try:
        subprocess_args = [sys.executable, '-m', 'sphinx',
                           '-b', 'html',
                           '-d', rst_doctree_dir,
                           '-D', f'headers_dest={c_header_dir}',
                           '-D', 'headers_mkdirs=0',
                           rst_source_dir,
                           rst_build_dir, ]
        if full_generation_flag:
            subprocess_args.append('-E')
        print("Executing sphinx in subprocess with args:", subprocess_args)
        return subprocess.run(subprocess_args).returncode
    except Exception:
        print('---')
        print('Have you installed sphinx?')
        print('---')
        raise


if __name__ == '__main__':
    sys.exit(run())
