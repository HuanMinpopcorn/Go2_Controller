import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock


from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config 
import numpy as np


class PhysicalSim:
    def __init__(self):
        self.locker = Lock()
        if config.ENABLE_CABLE_SCENE:
            self.mj_model = mujoco.MjModel.from_xml_path(config.CABLE_SCENE)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        self.elastic_band = None
        self.band_attached_link = None

        if config.ENABLE_ELASTIC_BAND:
            self.elastic_band = ElasticBand()
            if config.ROBOT == "h1" or config.ROBOT == "g1":
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.mj_model.opt.timestep = config.SIMULATE_DT
        self.num_motor_ = self.mj_model.nu
        self.dim_motor_sensor_ = 3 * self.num_motor_

        time.sleep(0.2)

    
        # Store foot trajectory
        self.reference_trajectory = []

        # Create an `mjvScene` object for rendering
        self.scene = mujoco.MjvScene(self.mj_model, maxgeom=1000)

    def simulation_thread(self):
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data)

        if config.USE_JOYSTICK:
            unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
        if config.PRINT_SCENE_INFORMATION:
            unitree.PrintSceneInformation()

        while self.viewer.is_running():
            step_start = time.perf_counter()

            self.locker.acquire()

            if config.ENABLE_ELASTIC_BAND:
                if self.elastic_band.enable:
                    self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                        self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                    )
            mujoco.mj_step(self.mj_model, self.mj_data)
            # self.update_foot_trajectory()

            self.locker.release()
            
            time_until_next_step = self.mj_model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


    def update_reference_trajectory(self, trajectory):
        """Update the reference trajectory visualization."""
        self.reference_trajectory = trajectory

    def update_foot_trajectory(self):
        """Visualizes only the reference trajectory using blue lines."""
        # Reset the number of geometries
        self.scene.ngeom = 0
        identity_mat = np.eye(3, dtype=np.float64).flatten()
        ref_line_color = np.array([0, 0, 1, 1], dtype=np.float32)  # Blue

        # Get the maximum number of geoms the scene can hold
        max_geoms = len(self.scene.geoms)

        for i in range(1, len(self.reference_trajectory)):
            if self.scene.ngeom >= max_geoms:
                break  # No more space to add geoms

            # Get the start and end points of the line segment
            start_point = self.reference_trajectory[i-1][:3].astype(np.float64)
            end_point = self.reference_trajectory[i][:3].astype(np.float64)

            # Access the current geom slot in the pre-allocated array
            geom = self.scene.geoms[self.scene.ngeom]

            # Initialize the geom as a line
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3, dtype=np.float64),  # Position (not used for line)
                np.zeros(3, dtype=np.float64),  # Direction (not used)
                identity_mat,
                ref_line_color
            )

            # Set the line endpoints using connector
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                0.002,  # Line width
                start_point,
                end_point
            )

            # Increment the geom counter
            self.scene.ngeom += 1

        # Update the scene to reflect the changes
        mujoco.mjv_updateScene(
            self.mj_model, self.mj_data, mujoco.MjvOption(),
            None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )

    def physics_viewer_thread(self):
        while self.viewer.is_running():
            self.locker.acquire()
            self.viewer.sync()
            self.locker.release()
            time.sleep(config.VIEWER_DT)

    def start(self):
        viewer_thread = Thread(target=self.physics_viewer_thread)
        sim_thread = Thread(target=self.simulation_thread)

        viewer_thread.start()
        sim_thread.start()


if __name__ == "__main__":
    sim = PhysicalSim()
    sim.start()
