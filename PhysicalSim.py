import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock


from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


class PhysicalSim:
    def __init__(self):
        self.locker = Lock()
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

            self.locker.release()

            time_until_next_step = self.mj_model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

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
