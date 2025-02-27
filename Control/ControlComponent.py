from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from interface.IOInterface import IOInterface
from interface.CmdPanel import CmdPanel
from common.unitreeRobot import QuadrupedRobot
from Gait.WaveGenerator import WaveGenerator
import numpy as np

# Optional debug import
try:
    from common.PyPlot import PyPlot
    COMPILE_DEBUG = True
except ImportError:
    COMPILE_DEBUG = False

class WaveStatus:
    STANCE_ALL = 'STANCE_ALL'
    SWING_ALL = 'SWING_ALL'
    WAVE_ALL = 'WAVE_ALL'

class CtrlComponents:
    def __init__(self, ioInter: IOInterface):
        self.ioInter = ioInter
        self.lowCmd = LowCmd_()
        self.lowState = LowState_()
        self.contact = np.zeros(4, dtype=int)
        self.phase = np.full(4, 0.5)
        self.robotModel = None
        self.waveGen = None
        self.estimator = None
        self.balCtrl = None
        self.dt = 0.0
        self.running = False
        self.ctrlPlatform = None
        self._waveStatus = WaveStatus.SWING_ALL
        if COMPILE_DEBUG:
            self.plot = PyPlot()
        else:
            self.plot = None

    def sendRecv(self):
        self.ioInter.sendRecv(self.lowCmd, self.lowState)

    def runWaveGen(self):
        self.waveGen.calcContactPhase(self.phase, self.contact, self._waveStatus)

    def setAllStance(self):
        self._waveStatus = WaveStatus.STANCE_ALL

    def setAllSwing(self):
        self._waveStatus = WaveStatus.SWING_ALL

    def setStartWave(self):
        self._waveStatus = WaveStatus.WAVE_ALL

    def geneObj(self):
        self.estimator = Estimator(self.robotModel, self.lowState, self.contact, self.phase, self.dt)
        self.balCtrl = BalanceCtrl(self.robotModel)
        if COMPILE_DEBUG and self.plot:
            self.balCtrl.setPyPlot(self.plot)
            self.estimator.setPyPlot(self.plot)
