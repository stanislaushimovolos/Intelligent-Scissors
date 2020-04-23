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
    def __init__(self, model, filename, fill_color="red", radius=5):
        self.model = model
        self.radius = radius
        self.fill_color = fill_color

        self.image = ImageTk.PhotoImage(Image.open(filename))
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)

    @property
    def canvas(self):
        return self.model.canvas

    def update(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)
        points = self.model.points
        for p in points:
            self.canvas.create_oval(
                p[0], p[1], p[0] + self.radius, p[1] + self.radius, fill=self.fill_color
            )


root = Tk()
stage = Canvas(root, bg="black", height=250, width=300)
stage.pack(expand=YES, fill=BOTH)

m = Model(stage)
m.add_view(v)

c = Controller(m)
stage.bind('<Button-1>', c.on_click)
root.mainloop()
