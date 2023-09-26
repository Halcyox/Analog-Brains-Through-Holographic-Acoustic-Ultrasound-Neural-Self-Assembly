import cupy as cp
import numpy as np
from mayavi import mlab
from traits.api import HasTraits, Range, Instance, Enum, on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

# Constants
c = 343  # Speed of sound in air (m/s)
f = 40e3  # Frequency (Hz)
wavelength = c / f

total_time = 1e-3  # Total simulation time (s)

# Spatial parameters
length = 0.04  # 4 cm
distance_between_transducers = 0.02  # 2 cm
dx = wavelength / 10
n_points = int(length / dx)  
n_time_steps = int(total_time / (dx / (2 * c)))
dt = dx / (2 * c)


# Gaussian pulse for simulating the acoustic pulse from the transducer
def gaussian(size, std):
    return cp.exp(-(cp.linspace(-size // 2, size // 2, size)**2) / (2 * std**2))

width = int(wavelength / dx)
gaussian_3d = gaussian(width, std=width//4).reshape(-1, 1, 1) * \
              gaussian(width, std=width//4).reshape(1, -1, 1) * \
              gaussian(width, std=width//4).reshape(1, 1, -1)

class Visualization(HasTraits):
    time_step = Range(0, n_time_steps-1, 0)
    pulse_type = Enum("Initial", "Continuous")
    scene = Instance(MlabSceneModel, ())
    title = "Holographic Assembly for Construction of Synthetic Nano-Brains"

    def __init__(self, **traits):
        super(Visualization, self).__init__(**traits)
        self.pressure_data = []
        self.run_simulation()

    def apply_boundary_condition(self, p):
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]

    def compute_gradient(self, arr):
        grad_x = (arr[2:, :, :] - arr[:-2, :, :]) / (2 * dx)
        grad_y = (arr[:, 2:, :] - arr[:, :-2, :]) / (2 * dx)
        grad_z = (arr[:, :, 2:] - arr[:, :, :-2]) / (2 * dx)
        return grad_x, grad_y, grad_z

    def run_simulation(self):
        p = cp.zeros((n_points, n_points, n_points))
        u = cp.zeros_like(p)
        v = cp.zeros_like(p)
        w = cp.zeros_like(p)

        mid_point = n_points // 2
        top_source = mid_point - int(distance_between_transducers / (2 * dx))
        bottom_source = mid_point + int(distance_between_transducers / (2 * dx))

        for t in range(n_time_steps):
            # Set up the sources
            if self.pulse_type == "Continuous" or (self.pulse_type == "Initial" and t == 0):
                p[top_source - width//2:top_source + width//2, 
                  mid_point - width//2:mid_point + width//2, 
                  mid_point - width//2:mid_point + width//2] += gaussian_3d

                p[bottom_source - width//2:bottom_source + width//2, 
                  mid_point - width//2:mid_point + width//2, 
                  mid_point - width//2:mid_point + width//2] += gaussian_3d

            # Compute the gradient of the pressure field
            grad_p_x, grad_p_y, grad_p_z = self.compute_gradient(p)

            # Update the velocity fields based on the gradient
            u[1:-1, :, :] += dt * grad_p_x
            v[:, 1:-1, :] += dt * grad_p_y
            w[:, :, 1:-1] += dt * grad_p_z

            # Compute the divergence of the velocity field
            div_u = (u[1:, :, :] - u[:-1, :, :]) / dx
            div_v = (v[:, 1:, :] - v[:, :-1, :]) / dx
            div_w = (w[:, :, 1:] - w[:, :, :-1]) / dx

            combined_div = (div_u[:-1, :-1, :-1] + div_v[:-1, :-1, :-1] + div_w[:-1, :-1, :-1])

            # Update pressure field based on the divergence
            p[1:-1, 1:-1, 1:-1] += dt * c**2 * combined_div

            # Apply boundary conditions to the pressure field
            self.apply_boundary_condition(p)

            # Store the pressure data for this time step
            self.pressure_data.append(cp.asnumpy(p))

    @on_trait_change('time_step,scene.activated')
    def update_plot(self):
        mlab.clf()
        pressure = self.pressure_data[self.time_step]
        mlab.contour3d(pressure, contours=10, transparent=True)

    view = View(Group(
                Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=600, width=800, show_label=False),
                Item(name='pulse_type'),
                Item(name='time_step'),
                ),
                resizable=True,
                title=title
                )

visualization = Visualization()
visualization.configure_traits()
