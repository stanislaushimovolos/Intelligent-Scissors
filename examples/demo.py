import argparse
from scissors.gui import run_demo


def main(file_name):
    run_demo(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', help='you, enter it')
    args = parser.parse_args()

    main(args.file_name)
