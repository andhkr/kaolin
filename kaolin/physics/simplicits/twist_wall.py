import torch
from functools import partial # Useful for boundary conditions

# Assume these are pre-loaded/defined SimplicitsObjects
# pipe_sim_obj = load_my_pipe_simplicit_object(...)

# Assume SimplicitsScene, SimulatedObject, SimplicitsObject etc. are imported
# from simplicits import SimplicitsScene, ...
# from simplicits import utils as simplicits_utils
# from simplicits import phys_utils
# from simplicits import optimization
from easy_api import *
import logging # Assuming logging setup is done elsewhere
logger = logging.getLogger(__name__)


def setup_twisted_pipe_scene(scene: SimplicitsScene, pipe_sim_obj: SimplicitsObject):
    """
    Sets up a scene with a pipe fixed at both ends.
    """
    print("Setting up twisted pipe scene (fixing both ends)...")

    # Add the pipe object to the scene
    obj_idx = scene.add_object(pipe_sim_obj)

    # --- Define Boundary Conditions ---
    # We need functions that identify points at each end based on rest positions
    # IMPORTANT: Adjust these functions based on your pipe's orientation and dimensions!
    # Let's assume the pipe lies mostly along the Z-axis.

    pipe_pts_rest = pipe_sim_obj.pts # Get rest vertices (N, 3)
    z_coords = pipe_pts_rest[:, 2]
    min_z, max_z = torch.min(z_coords), torch.max(z_coords)
    pipe_length = max_z - min_z
    end_threshold = pipe_length * 0.05 # Consider points within 5% of ends

    def identify_pipe_end1(pts_rest):
        # Identify points at the 'minimum Z' end
        return pts_rest[:, 2] < (min_z + end_threshold)

    def identify_pipe_end2(pts_rest):
        # Identify points at the 'maximum Z' end
        return pts_rest[:, 2] > (max_z - end_threshold)

    # Apply the boundary conditions to fix the points identified by the functions
    # High penalty means the points will be strongly fixed.
    fix_penalty = 1e6
    scene.set_object_boundary_condition(obj_idx, "fix_end1", identify_pipe_end1, fix_penalty)
    scene.set_object_boundary_condition(obj_idx, "fix_end2", identify_pipe_end2, fix_penalty)
    print(f"Applied boundary conditions to fix ends (Z < {min_z + end_threshold:.3f} and Z > {max_z - end_threshold:.3f})")

    # --- Set Material Properties ---
    # Make it reasonably stiff so it doesn't just collapse
    scene.set_object_materials(obj_idx, yms=torch.tensor(5e4, device=scene.default_device, dtype=scene.default_dtype))
    print("Set pipe material properties (Yms=5e4).")

    # --- Optional: Set Gravity ---
    # You might add slight gravity to see some deformation if the pipe isn't perfectly straight
    # scene.set_scene_gravity(acc_gravity=torch.tensor([0, -0.1, 0], device=scene.default_device))
    # print("Set mild gravity.")

    return obj_idx


import torch
from functools import partial

# Assume these are pre-loaded/defined SimplicitsObjects
# bar_sim_obj = load_my_bar_simplicit_object(...)

# Assume SimplicitsScene, SimulatedObject, SimplicitsObject etc. are imported

def setup_cantilever_bar_scene(scene: SimplicitsScene, bar_sim_obj: SimplicitsObject):
    """
    Sets up a scene with a bar fixed at one end, subject to gravity.
    """
    print("Setting up cantilever bar scene...")

    # Add the bar object to the scene
    obj_idx = scene.add_object(bar_sim_obj)

    # --- Define Boundary Condition ---
    # IMPORTANT: Adjust this function based on your bar's orientation and dimensions!
    # Let's assume the bar lies mostly along the X-axis and we fix the 'minimum X' end.

    bar_pts_rest = bar_sim_obj.pts # Get rest vertices (N, 3)
    x_coords = bar_pts_rest[:, 0]
    min_x, max_x = torch.min(x_coords), torch.max(x_coords)
    bar_length = max_x - min_x
    end_threshold = bar_length * 0.05 # Fix points within 5% of one end

    def identify_bar_fixed_end(pts_rest):
        # Identify points at the 'minimum X' end
        return pts_rest[:, 0] < (min_x + end_threshold)

    # Apply the boundary condition to fix the points identified by the function
    fix_penalty = 1e6
    scene.set_object_boundary_condition(obj_idx, "fix_wall_end", identify_bar_fixed_end, fix_penalty)
    print(f"Applied boundary condition to fix end (X < {min_x + end_threshold:.3f})")

    # --- Set Material Properties ---
    # Make it flexible like rubber (lower Young's Modulus 'yms')
    # Also set density ('rhos') so gravity has a noticeable effect
    scene.set_object_materials(obj_idx,
                               yms=torch.tensor(1e4, device=scene.default_device, dtype=scene.default_dtype), # Lower stiffness
                               rhos=torch.tensor(1000, device=scene.default_device, dtype=scene.default_dtype)) # Density (e.g., kg/m^3)
    print("Set bar material properties (Yms=1e4, Rhos=1000).")

    # --- Set Gravity ---
    # Apply gravity pulling downwards along the Y-axis (standard convention)
    # Adjust the direction if your coordinate system is different (e.g., Z-down)
    gravity_vector = torch.tensor([0.0, -9.8, 0.0], device=scene.default_device, dtype=scene.default_dtype)
    scene.set_scene_gravity(acc_gravity=gravity_vector)
    print(f"Set scene gravity: {gravity_vector.cpu().numpy()}")

    # --- Optional: Set Floor ---
    # Add a floor below the bar to prevent it falling indefinitely
    scene.set_scene_floor(floor_height=-0.8, floor_axis=1, floor_penalty=10000) # Floor at y = -0.8
    print("Set scene floor at Y = -0.8.")

    return obj_idx