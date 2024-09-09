"""
Script to clean up checkpoints from a run if want to go back and reduce disk space used.
"""

import argparse
import re
import os
from os import path as osp
import shutil

# This regex will match the checkpoint file names we use and extract the sequence number (so that we can keep last ones).
REGEX = re.compile(r"checkpoint_(\d+)")
NUM_TO_KEEP = 2


def clean_up_below_folder(path, get_dirs_to_remove, dry_run=False):
    for root, dirnames, fnames in os.walk(path):
        ds_to_del = get_dirs_to_remove(dirnames)
        for d in dirnames:
            if d in ds_to_del:
                pth_to_remove = osp.join(root, d)
                if not dry_run:
                    shutil.rmtree(pth_to_remove)
                    print(f"removed {pth_to_remove}")
                else:
                    print(f"would remove {pth_to_remove}")
            else:
                clean_up_below_folder(osp.join(root, d), get_dirs_to_remove, dry_run)


def get_dirs_to_remove(all_dirs):
    """will return a list of all dirs that match the regex and are not the last `NUM_TO_KEEP`"""
    matches = [REGEX.match(el) for el in all_dirs]
    matches = [el for el in matches if el is not None]
    numbers_ = [int(el.group(1)) for el in matches]
    numbers_to_keep = set(sorted(numbers_)[-NUM_TO_KEEP:])

    to_remove = [el.group(0) for el in matches if not int(el.group(1)) in numbers_to_keep]
    return to_remove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    clean_up_below_folder(args.path, get_dirs_to_remove, args.dry_run)


if __name__ == '__main__':
    main()
