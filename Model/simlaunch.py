import mujoco_py
import numpy as np

# Load the MuJoCo model
model = mujoco_py.load_model_from_path("cable.xml")
sim = mujoco_py.MjSim(model)

# Create a viewer (to see the simulation)
viewer = mujoco_py.MjViewer(sim)

# Set a control signal to move Box 1
sim.data.ctrl[0] = 0.05  # Control input for motor, can be adjusted between -0.1 and 0.1

# Run the simulation loop
while True:
    sim.step()  # Advance the simulation
    viewer.render()  # Render the simulation
