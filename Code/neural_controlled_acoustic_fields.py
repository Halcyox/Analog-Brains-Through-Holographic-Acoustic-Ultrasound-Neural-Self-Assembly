import cupy as cp
import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Range, Instance, on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

# Define parameters
length = 0.05  # domain side length in meters
total_time = 1e-3  # total simulation time in seconds
c = 343  # speed of sound in air in m/s

n_points = 100  # Number of discretization points along one dimension
dx = length / n_points  # Spatial step
dt = dx / (2 * c)  # Time step based on CFL condition
n_time_steps = int(total_time / dt)

# Gaussian pulse
def gaussian(size, std):
    return cp.exp(-(cp.linspace(-size // 2, size // 2, size)**2) / (2 * std**2))

width = n_points // 10  # Width of the Gaussian pulse
gaussian_3d = gaussian(width, std=width//4).reshape(-1, 1, 1) * \
              gaussian(width, std=width//4).reshape(1, -1, 1) * \
              gaussian(width, std=width//4).reshape(1, 1, -1)


class Visualization(HasTraits):
    time_step = Range(0, n_time_steps-1, 0)
    scene = Instance(MlabSceneModel, ())

    # Pressure data storage
    pressure_data = []

    def __init__(self):
        super(Visualization, self).__init__()
        self.run_simulation()
    
    def run_simulation(self):
        p = cp.zeros((n_points, n_points, n_points))
        p_prev = cp.zeros_like(p)
        p_next = cp.zeros_like(p)

        mid_point = n_points // 2
        start_point = mid_point - width // 2
        end_point = mid_point + width // 2
        p[start_point:end_point, start_point:end_point, start_point:end_point] = gaussian_3d

        for t in range(n_time_steps):
            laplacian = (cp.roll(p, -1, axis=0) + cp.roll(p, 1, axis=0) - 2*p) / dx**2 + \
                        (cp.roll(p, -1, axis=1) + cp.roll(p, 1, axis=1) - 2*p) / dx**2 + \
                        (cp.roll(p, -1, axis=2) + cp.roll(p, 1, axis=2) - 2*p) / dx**2
            p_next = 2*p - p_prev + c**2 * dt**2 * laplacian
            self.pressure_data.append(cp.asnumpy(p))
            p_prev, p, p_next = p, p_next, p_prev

    @on_trait_change('time_step,scene.activated')
    def update_plot(self):
        mlab.clf()
        data = np.array(self.pressure_data[self.time_step], dtype=np.float32)
        s = mlab.pipeline.scalar_field(data)
        mlab.pipeline.iso_surface(s, opacity=0.5)
        mlab.colorbar()
        mlab.title(f"Time step: {self.time_step}")


    # The layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=600, width=800, show_label=False),
                Group('_', 'time_step'),
                resizable=True)

visualization = Visualization()
visualization.configure_traits()
