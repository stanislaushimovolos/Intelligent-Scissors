import skimage
import argparse
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

from scissors.graph import PathFinder
from scissors.gui import Poly, PolyView, PolyController, Pixels, PixelsView
from scissors.feature_extraction import StaticExtractor, DynamicExtractor, Scissors


class GuiManager:
    def __init__(self, canvas, scissors):
        self.canvas = canvas
        self.scissors = scissors

        self.pixel_model = Pixels(self.canvas)
        self.pixel_view = PixelsView(self.pixel_model)
        self.pixel_model.add_view(self.pixel_view)

        self.poly_model = Poly(self.canvas)
        self.poly_view = PolyView(self.poly_model)
        self.poly_model.add_view(self.poly_view)

        self.c = PolyController(self.poly_model)
        self.prev_click = None
        self.cur_click = None

    def on_click(self, e):
        self.prev_click = self.cur_click
        self.cur_click = np.flip((e.x, e.y))
        self.c.on_click(e)

        if self.prev_click is not None:
            path = self.scissors.find_path(self.prev_click, self.cur_click)
            path = [np.flip(x) for x in path]
            self.pixel_model.add_pixels(path)


def main(file_name):
    image = Image.open(file_name)
    w, h = image.size
    gray_scaled = skimage.color.rgb2gray(np.asarray(image))

    static_extractor = StaticExtractor()
    static_cost = static_extractor(gray_scaled)
    finder = PathFinder(image.size, np.squeeze(static_cost))

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
