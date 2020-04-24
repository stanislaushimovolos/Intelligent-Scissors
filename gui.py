from tkinter import *
from PIL import ImageTk, Image


class Model:
    def __init__(self, canvas):
        self.canvas = canvas
        self.points = []
        self.views = []

    def add_view(self, view):
        self.views.append(view)

    def add_point(self, point):
        self.points.append(point)
        self.update()

    def update(self):
        for view in self.views:
            view.update()


class Controller:
    def __init__(self, model):
        self.model = model

    def on_click(self, event):
        self.model.add_point((event.x, event.y))

    @property
    def canvas(self):
        return self.model.canvas


class View:
    def __init__(self, model, image, fill_color="red", radius=3):
        self.model = model

        self.radius = radius
        self.fill_color = fill_color

        self.image = ImageTk.PhotoImage(image)
        self.rectangles = []
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)

    def update(self):
        self.canvas.delete("all")
        # print(self.canvas['width'])
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)
        points = self.model.points

        # for previous, current in zip(points, points[1:]):
        #   self.canvas.create_line(*previous, *current, fill=self.fill_color)

        for p in points:
            self.canvas.create_oval(
                p[0] - self.radius, p[1] - self.radius,
                p[0] + self.radius, p[1] + self.radius,
                fill=self.fill_color
            )
        for r in self.rectangles:
            x, y = r
            self.canvas.create_rectangle((x, y) * 2, outline="red")

    @property
    def canvas(self):
        return self.model.canvas
