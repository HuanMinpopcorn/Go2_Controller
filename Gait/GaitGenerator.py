import numpy as np
from FeetEndCal import FeetEndCal

class GaitGenerator:
    def __init__(self, ctrlComp):
        self._waveG = ctrlComp.waveGen
        self._est = ctrlComp.estimator
        self._phase = ctrlComp.phase
        self._contact = ctrlComp.contact
        self._robModel = ctrlComp.robotModel
        self._state = ctrlComp.lowState

        self._feetCal = FeetEndCal(ctrlComp)
        self._firstRun = True

    def setGait(self, vxyGoalGlobal, dYawGoal, gaitHeight):
        self._vxyGoal = vxyGoalGlobal
        self._dYawGoal = dYawGoal
        self._gaitHeight = gaitHeight

    def restart(self):
        self._firstRun = True
        self._vxyGoal = np.zeros(2)

    def run(self, feetPos, feetVel):
        if self._firstRun:
            self._startP = self._est.getFeetPos()
            self._firstRun = False

        for i in range(4):
            if self._contact[i] == 1:
                if self._phase[i] < 0.5:
                    self._startP[:, i] = self._est.getFootPos(i)
                feetPos[:, i] = self._startP[:, i]
                feetVel[:, i] = np.zeros(3)
            else:
                self._endP[:, i] = self._feetCal.calFootPos(i, self._vxyGoal, self._dYawGoal, self._phase[i])

                feetPos[:, i] = self.getFootPos(i)
                feetVel[:, i] = self.getFootVel(i)

        self._pastP = feetPos.copy()
        self._phasePast = self._phase.copy()

    def getFootPos(self, i):
        footPos = np.zeros(3)

        footPos[0] = self.cycloidXYPosition(self._startP[0, i], self._endP[0, i], self._phase[i])
        footPos[1] = self.cycloidXYPosition(self._startP[1, i], self._endP[1, i], self._phase[i])
        footPos[2] = self.cycloidZPosition(self._startP[2, i], self._gaitHeight, self._phase[i])

        return footPos

    def getFootVel(self, i):
        footVel = np.zeros(3)

        footVel[0] = self.cycloidXYVelocity(self._startP[0, i], self._endP[0, i], self._phase[i])
        footVel[1] = self.cycloidXYVelocity(self._startP[1, i], self._endP[1, i], self._phase[i])
        footVel[2] = self.cycloidZVelocity(self._gaitHeight, self._phase[i])

        return footVel

    def cycloidXYPosition(self, start, end, phase):
        phasePI = 2 * np.pi * phase
        return (end - start) * (phasePI - np.sin(phasePI)) / (2 * np.pi) + start

    def cycloidXYVelocity(self, start, end, phase):
        phasePI = 2 * np.pi * phase
        return (end - start) * (1 - np.cos(phasePI)) / self._waveG.getTswing()

    def cycloidZPosition(self, start, h, phase):
        phasePI = 2 * np.pi * phase
        return h * (1 - np.cos(phasePI)) / 2 + start

    def cycloidZVelocity(self, h, phase):
        phasePI = 2 * np.pi * phase
        return h * np.pi * np.sin(phasePI) / self._waveG.getTswing()
