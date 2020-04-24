import numpy as np
import skimage
from tkinter import *
from PIL import ImageTk, Image

from graph import PathFinder
from feature_extraction import StaticFeatureExtractor
from gui import Poly, PolyView, PolyController, Pixels, PixelsView


class ScissorManager:
    def __init__(self, canvas, image):
        self.canvas = canvas
        gray_scaled = skimage.color.rgb2gray(np.asarray(image))

        self.extractor = StaticFeatureExtractor()
        self.cost = self.extractor.get_total_link_costs(gray_scaled)
        self.finder = PathFinder(image.size, np.squeeze(self.cost))

        self.pixel_model = Pixels(self.canvas)
        self.pixel_view = PixelsView(self.pixel_model)
        self.pixel_model.add_view(self.pixel_view)

        self.poly_model = Poly(self.canvas)
        self.poly_view = PolyView(self.poly_model)
        self.poly_model.add_view(self.poly_view)

        self.c = PolyController(self.poly_model)
        self.clicks = []

    def on_click(self, event):
        self.clicks.append((event.x, event.y))
        self.c.on_click(event)
        if len(self.clicks) > 1:
            coord = self.finder.find_path(np.flip(self.clicks[-2]), (np.flip(self.clicks[-1])))
            coord = [np.flip(cor) for cor in coord]
            self.pixel_model.add_pixels(coord)


def main():
    root = Tk()
    image = Image.open('392.png')
    h, w = image.size

    stage = Canvas(root, bg="black", width=w, height=h)
    tk_image = ImageTk.PhotoImage(image)
    stage.create_image(0, 0, image= tk_image, anchor=NW)

    scissors = ScissorManager(stage, image)
    stage.bind('<Button-1>', scissors.on_click)
    stage.pack(expand=YES, fill=BOTH)

    root.resizable(False, False)
    root.mainloop()


if __name__ == '__main__':
    main()
