import cupy as cp
import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Range, Instance, Enum, on_trait_change, Str
from traitsui.api import View, Item, Group, Label
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

# Define parameters
length = 0.05  # domain side length in meters
total_time = 1e-3  # total simulation time in seconds
c = 343  # speed of sound in air in m/s
B = 1e-9  # Non-linearity parameter (this is a dummy value, the actual value should be determined experimentally or from literature)

n_points = 100  # Number of discretization points along one dimension
dx = length / n_points  # Spatial step
dt = dx / (2 * c)  # Time step based on CFL condition
n_time_steps = int(total_time / dt)

mid_point = n_points // 2  # Definition of midpoint

# Gaussian pulse
def gaussian(size, std):
    return cp.exp(-(cp.linspace(-size // 2, size // 2, size)**2) / (2 * std**2))

width = n_points // 10  # Width of the Gaussian pulse

gaussian_3d = gaussian(width, std=width//4).reshape(-1, 1, 1) * \
              gaussian(width, std=width//4).reshape(1, -1, 1) * \
              gaussian(width, std=width//4).reshape(1, 1, -1)

class Visualization(HasTraits):
    time_step = Range(0, n_time_steps-1, 0)
    setup_type = Enum("Center", "Diamond", "Cube", "Ring")
    scene = Instance(MlabSceneModel, ())
    title = Str("Holographic Assembly for Construction of Synthetic Nano-Brains")

    # New method to apply absorbing boundary condition
    def apply_boundary_condition(self, p):
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]
        return p

    # Pressure data storage
    pressure_data = []

    def __init__(self):
        super(Visualization, self).__init__()
        self.run_simulation()

    # New method for gradient computation
    def compute_gradient(self, arr):
        grad_x = (arr[2:, 1:-1, 1:-1] - arr[:-2, 1:-1, 1:-1]) / (2 * dx)
        grad_y = (arr[1:-1, 2:, 1:-1] - arr[1:-1, :-2, 1:-1]) / (2 * dx)
        grad_z = (arr[1:-1, 1:-1, 2:] - arr[1:-1, 1:-1, :-2]) / (2 * dx)
        return grad_x, grad_y, grad_z
    
    @on_trait_change('setup_type')
    def run_simulation(self):
        # Reset pressure data and time step
        self.pressure_data = []
        self.time_step = 0

        p = cp.zeros((n_points, n_points, n_points))
        p_prev = cp.zeros_like(p)
        p_next = cp.zeros_like(p)

        # Define 6 points for a 3D diamond
        # Initial setups
        if self.setup_type == "Center":
            points = [(mid_point, mid_point, mid_point)]
        elif self.setup_type == "Diamond":
            points = [
                (mid_point, mid_point, mid_point - width),
                (mid_point, mid_point, mid_point + width),
                (mid_point - width, mid_point, mid_point),
                (mid_point + width, mid_point, mid_point),
                (mid_point, mid_point - width, mid_point),
                (mid_point, mid_point + width, mid_point)
            ]
        elif self.setup_type == "Cube":
            d = width // 2
            points = [
                (mid_point - d, mid_point - d, mid_point - d),
                (mid_point - d, mid_point - d, mid_point + d),
                (mid_point - d, mid_point + d, mid_point - d),
                (mid_point - d, mid_point + d, mid_point + d),
                (mid_point + d, mid_point - d, mid_point - d),
                (mid_point + d, mid_point - d, mid_point + d),
                (mid_point + d, mid_point + d, mid_point - d),
                (mid_point + d, mid_point + d, mid_point + d),
            ]
        elif self.setup_type == "Ring":
            r = width // 2
            theta = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 8 points around a circle
            points = [(int(round(mid_point + r * np.cos(t))), int(round(mid_point + r * np.sin(t))), mid_point) for t in theta]


        # Introduce Gaussian pulse at each point
        for point in points:
            x, y, z = point
            start_x, start_y, start_z = x - width // 2, y - width // 2, z - width // 2
            end_x, end_y, end_z = x + width // 2, y + width // 2, z + width // 2
            p[start_x:end_x, start_y:end_y, start_z:end_z] += gaussian_3d

        for t in range(n_time_steps):
            laplacian = (cp.roll(p, -1, axis=0) + cp.roll(p, 1, axis=0) - 2*p) / dx**2 + \
                        (cp.roll(p, -1, axis=1) + cp.roll(p, 1, axis=1) - 2*p) / dx**2 + \
                        (cp.roll(p, -1, axis=2) + cp.roll(p, 1, axis=2) - 2*p) / dx**2

            grad_x, grad_y, grad_z = self.compute_gradient(p**2)
            nonlinear_term = B * (grad_x + grad_y + grad_z)

            # Pad the nonlinear term to match the shape of laplacian
            nonlinear_term = cp.pad(nonlinear_term, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

            p_next = 2*p - p_prev + c**2 * dt**2 * (laplacian + nonlinear_term)
            p_next = self.apply_boundary_condition(p_next)
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
    view = View(Group(
                    Item('title', show_label=False, style='readonly'),
                    Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                        height=600, width=800, show_label=False),
                    Item('setup_type', label="Initial Setup"),
                    '_',
                    'time_step',
                    orientation="vertical"
                ),
                resizable=True,
                title="Nano-Brain Holographic Assembly"
                )

visualization = Visualization()
visualization.configure_traits()
