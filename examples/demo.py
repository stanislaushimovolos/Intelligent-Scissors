import skimage
import argparse
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

from scissors.gui import GuiManager
from scissors.graph import PathFinder
from scissors.feature_extraction import StaticExtractor, DynamicExtractor, Scissors


def main(file_name):
    image = Image.open(file_name)
    w, h = image.size
    gray_scaled = skimage.color.rgb2gray(np.asarray(image))

    static_extractor = StaticExtractor()
    static_cost = static_extractor(gray_scaled)
    finder = PathFinder(image.size, static_cost)

    dynamic_extractor = DynamicExtractor()
    dynamic_features = dynamic_extractor(gray_scaled)
    scissors = Scissors(static_cost, dynamic_features, finder)

    root = Tk()
    stage = Canvas(root, bg="black", width=w, height=h)
    tk_image = ImageTk.PhotoImage(image)
    stage.create_image(0, 0, image=tk_image, anchor=NW)

    manager = GuiManager(stage, scissors)
    stage.bind('<Button-1>', manager.on_click)

    stage.pack(expand=YES, fill=BOTH)
    root.resizable(False, False)
    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', help='your name, enter it')
    args = parser.parse_args()

    main(args.file_name)
