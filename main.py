import argparse
import os

from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import *
from photo_adjuster import adjust_photo


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default='input/good/easy3.jpg'
    )

    return parser.parse_args()

def clear_output_directory(directory='output'):
    """
    Clears all images in the specified directory.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.unlink(file_path)

def main():
    clear_output_directory()
    args = parse()
    image = cv2.imread(args.input)
    adjusted_photo = adjust_photo(image)
    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = extract_notes(blobs, staffs, adjusted_photo)
    draw_notes_pitch(adjusted_photo, notes)


if __name__ == "__main__":
    main()
