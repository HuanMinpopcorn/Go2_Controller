import numpy as np

class FeetEndCal:
    def __init__(self, ctrlComp):
        self._est = ctrlComp.estimator
        self._lowState = ctrlComp.lowState
        self._robModel = ctrlComp.robotModel
        
        self._Tstance = ctrlComp.waveGen.getTstance()
        self._Tswing = ctrlComp.waveGen.getTswing()

        self._kx = 0.005
        self._ky = 0.005
        self._kyaw = 0.005

        feetPosBody = self._robModel.getFeetPosIdeal()
        self._feetRadius = np.zeros(4)
        self._feetInitAngle = np.zeros(4)
        for i in range(4):
            self._feetRadius[i] = np.sqrt(feetPosBody[0, i]**2 + feetPosBody[1, i]**2)
            self._feetInitAngle[i] = np.arctan2(feetPosBody[1, i], feetPosBody[0, i])

    def calFootPos(self, legID, vxyGoalGlobal, dYawGoal, phase):
        self._bodyVelGlobal = self._est.getVelocity()
        self._bodyWGlobal = self._lowState.getGyroGlobal()

        self._nextStep = np.zeros(3)
        self._nextStep[0] = (self._bodyVelGlobal[0] * (1 - phase) * self._Tswing +
                              self._bodyVelGlobal[0] * self._Tstance / 2 +
                              self._kx * (self._bodyVelGlobal[0] - vxyGoalGlobal[0]))
        self._nextStep[1] = (self._bodyVelGlobal[1] * (1 - phase) * self._Tswing +
                              self._bodyVelGlobal[1] * self._Tstance / 2 +
                              self._ky * (self._bodyVelGlobal[1] - vxyGoalGlobal[1]))
        self._nextStep[2] = 0

        self._yaw = self._lowState.getYaw()
        self._dYaw = self._lowState.getDYaw()
        self._nextYaw = (self._dYaw * (1 - phase) * self._Tswing +
                         self._dYaw * self._Tstance / 2 +
                         self._kyaw * (dYawGoal - self._dYaw))

        self._nextStep[0] += (self._feetRadius[legID] *
                               np.cos(self._yaw + self._feetInitAngle[legID] + self._nextYaw))
        self._nextStep[1] += (self._feetRadius[legID] *
                               np.sin(self._yaw + self._feetInitAngle[legID] + self._nextYaw))

        self._footPos = self._est.getPosition() + self._nextStep
        self._footPos[2] = 0.0

        return self._footPos
