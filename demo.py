import numpy as np
import skimage
from tkinter import *
from PIL import Image

from feature_extraction import StaticFeatureExtractor
from graph import PathFinder
from gui import Model, View, Controller


class ScissorsController:
    def __init__(self):
        pass


def main():
    image = Image.open('test_img.jpg')
    width, height = image.size

    img = np.asarray(image)
    img = skimage.color.rgb2gray(img)
    extractor = StaticFeatureExtractor()
    cost = extractor.get_total_link_costs(img)

    finder = PathFinder(img.shape, np.squeeze(cost))

    root = Tk()
    stage = Canvas(root, bg="black", height=height, width=width)
    stage.pack(expand=YES, fill=BOTH)

    m = Model(stage)
    v = View(m, image=image)
    m.add_view(v)

    clicks = []

    c = Controller(m)

    def on_click(event):
        clicks.append((event.x, event.y))
        c.on_click(event)
        if len(clicks) > 1:
            coord = finder.find_path(np.flip(clicks[-2]), (np.flip(clicks[-1])))
            for cor in coord:
                v.rectangles.append(np.flip(cor))
            v.update()

    stage.bind('<Button-1>', on_click)

    root.resizable(False, False)
    root.mainloop()


if __name__ == '__main__':
    main()
