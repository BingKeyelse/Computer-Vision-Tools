class MouseTool:
    def on_mouse_down(self, event):
        pass
    
    def on_mouse_move(self, event):
        pass
    
    def on_mouse_up(self, event):
        pass
    
    def draw(self, canvas):
        pass

class BoxTool(MouseTool):
    def __init__(self):
        self.start = None
        self.end = None

    def on_mouse_down(self, event):
        self.start = (event.x, event.y)

    def on_mouse_move(self, event):
        if self.start:
            self.end = (event.x, event.y)

    def on_mouse_up(self, event):
        self.end = (event.x, event.y)

    def draw(self, canvas):
        if self.start and self.end:
            canvas.draw_rect(self.start, self.end)


class PolygonTool(MouseTool):
    def __init__(self):
        self.points = []

    def on_mouse_down(self, event):
        self.points.append((event.x, event.y))

    def draw(self, canvas):
        if len(self.points) > 1:
            canvas.draw_polygon(self.points)

class ToolManager:
    def __init__(self):
        self.active_tool: MouseTool | None = None

    def set_tool(self, tool: MouseTool):
        self.active_tool = tool

    def handle_event(self, event_type, event):
        if self.active_tool:
            getattr(self.active_tool, f"on_{event_type}")(event)

    def draw(self, canvas):
        if self.active_tool:
            self.active_tool.draw(canvas)
