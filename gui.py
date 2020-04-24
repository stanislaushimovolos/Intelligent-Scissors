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
    def __init__(self, model, fill_color="red"):
        super().__init__(model)
        self.fill_color = fill_color

    def update(self):
        pixels = self.model.pixels

        for pix in pixels:
            x, y = pix
            self.canvas.create_rectangle((x, y) * 2, outline=self.fill_color)


class PolyView(View):
    def __init__(self, model, draw_lines=False, fill_color="red", radius=3):
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
