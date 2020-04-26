import numpy as np


class Model:
    def __init__(self, canvas):
        self.canvas = canvas
        self.views = []

    def add_view(self, view):
        self.views.append(view)

    def update(self):
        for view in self.views:
            view.update()


class View:
    def __init__(self, model):
        self.model = model

    def update(self):
        raise NotImplementedError()

    @property
    def canvas(self):
        return self.model.canvas


class Poly(Model):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.points = []

    def add_point(self, point):
        self.points.append(point)
        self.update()


class Pixels(Model):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.pixels = []

    def add_pixels(self, pixels):
        self.pixels.extend(pixels)
        self.update()


class PolyController:
    def __init__(self, model):
        self.model = model

    def on_click(self, event):
        self.model.add_point((event.x, event.y))

    @property
    def canvas(self):
        return self.model.canvas


class PixelsView(View):
    def __init__(self, model, fill_color="blue"):
        super().__init__(model)
        self.fill_color = fill_color

    def update(self):
        pixels = self.model.pixels

        for pix in pixels:
            x, y = pix
            self.canvas.create_rectangle((x, y) * 2, outline=self.fill_color)


class PolyView(View):
    def __init__(self, model, draw_lines=False, fill_color="green", radius=3):
        super().__init__(model)

        self.radius = radius
        self.fill_color = fill_color
        self.draw_lines = draw_lines

    def update(self):
        points = self.model.points

        if self.draw_lines:
            for previous, current in zip(points, points[1:]):
                self.canvas.create_line(*previous, *current, fill=self.fill_color)

        for p in points:
            self.canvas.create_oval(
                p[0] - self.radius, p[1] - self.radius,
                p[0] + self.radius, p[1] + self.radius,
                fill=self.fill_color
            )


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
