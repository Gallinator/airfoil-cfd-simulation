import bezier.curve
import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance from the point *p* to the segment (*s0*, *s1*), where
    *p*, *s0*, *s1* are ``[x, y]`` arrays.
    """
    s01 = s1 - s0
    s0p = p - s0
    if (s01 == 0).all():
        return np.hypot(*s0p)
    # Project onto segment, without going past segment ends.
    p1 = s0 + np.clip((s0p @ s01) / (s01 @ s01), 0, 1) * s01
    return np.hypot(*(p - p1))


class AirfoilInteractor:
    """
    An airfoil editor. Draws an upper and lower bezier. The middle leading and trailing nodes are fixed while all the leading nodes can ba dragged only on the y axis.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    """

    showverts = True
    epsilon = 10  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.poly.set_visible(False)

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', color='#00000000', animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

        self.bezier_sampling_points = np.linspace(0.0, 1.0, 100)
        self.airfoil_poly = Polygon(self.get_airfoil_poly_xy(), animated=True, facecolor='lightgrey')
        self.ax.add_patch(self.airfoil_poly)

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.airfoil_poly)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self._ind == 0 or self._ind == (len(self.poly.xy) - 1) / 2:
            return

        x, y = event.xdata, event.ydata
        prev_x, _ = self.poly.xy[self._ind]

        if self._ind == 1 or self._ind == len(self.poly.xy) - 2:
            x = prev_x

        self.poly.xy[self._ind] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.airfoil_poly.set_xy(self.get_airfoil_poly_xy())

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.airfoil_poly)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def get_airfoil_poly_xy(self):
        bezier_points = self.get_bezier_points(self.bezier_sampling_points, self.bezier_sampling_points)
        return np.concatenate(bezier_points, 1).T

    def get_bezier_points(self, sampling_points_upper: np.ndarray, sampling_points_lower: np.ndarray):
        nodes_x, nodes_y = zip(*self.poly.xy)
        trailing_idx = len(self.poly.xy) // 2

        upper_x, upper_y = nodes_x[0:trailing_idx + 1], nodes_y[0:trailing_idx + 1]
        lower_x, lower_y = nodes_x[trailing_idx:], nodes_y[trailing_idx:]

        upper_curve = bezier.curve.Curve.from_nodes([upper_x, upper_y])
        lower_curve = bezier.curve.Curve.from_nodes([lower_x, lower_y])
        return upper_curve.evaluate_multi(sampling_points_upper), lower_curve.evaluate_multi(sampling_points_lower)
