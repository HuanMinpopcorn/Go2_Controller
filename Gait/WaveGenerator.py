import time
import numpy as np

class WaveStatus:
    WAVE_ALL = 'WAVE_ALL'
    SWING_ALL = 'SWING_ALL'
    STANCE_ALL = 'STANCE_ALL'

class WaveGenerator:
    def __init__(self, period, stance_phase_ratio, bias):
        self._period = period
        self._st_ratio = stance_phase_ratio
        self._bias = np.array(bias)

        if not (0 < self._st_ratio < 1):
            raise ValueError("[ERROR] The stancePhaseRatio of WaveGenerator should be between (0, 1)")

        if np.any((self._bias < 0) | (self._bias > 1)):
            raise ValueError("[ERROR] The bias of WaveGenerator should be between [0, 1]")

        self._start_time = time.time()
        self._contact_past = np.zeros(4, dtype=int)
        self._phase_past = np.full(4, 0.5)
        self._status_past = WaveStatus.SWING_ALL
        self._switch_status = np.ones(4, dtype=int)

    def calc_contact_phase(self, phase_result, contact_result, status):
        self.calc_wave(status)

        if status != self._status_past:
            if np.sum(self._switch_status) == 0:
                self._switch_status = np.ones(4, dtype=int)
            self.calc_wave(self._status_past)

            if status == WaveStatus.STANCE_ALL and self._status_past == WaveStatus.SWING_ALL:
                self._contact_past = np.ones(4, dtype=int)
            elif status == WaveStatus.SWING_ALL and self._status_past == WaveStatus.STANCE_ALL:
                self._contact_past = np.zeros(4, dtype=int)

        if np.sum(self._switch_status) != 0:
            for i in range(4):
                if self._contact[i] == self._contact_past[i]:
                    self._switch_status[i] = 0
                else:
                    self._contact[i] = self._contact_past[i]
                    self._phase[i] = self._phase_past[i]

            if np.sum(self._switch_status) == 0:
                self._status_past = status

        phase_result[:] = self._phase
        contact_result[:] = self._contact

    def get_t_stance(self):
        return self._period * self._st_ratio

    def get_t_swing(self):
        return self._period * (1 - self._st_ratio)

    def get_t(self):
        return self._period

    def calc_wave(self, status):
        self._phase = np.zeros(4)
        self._contact = np.zeros(4, dtype=int)
        if status == WaveStatus.WAVE_ALL:
            pass_time = (time.time() - self._start_time)
            for i in range(4):
                normal_t = (pass_time + self._period - self._period * self._bias[i]) % self._period / self._period
                if normal_t < self._st_ratio:
                    self._contact[i] = 1
                    self._phase[i] = normal_t / self._st_ratio
                else:
                    self._contact[i] = 0
                    self._phase[i] = (normal_t - self._st_ratio) / (1 - self._st_ratio)

        elif status == WaveStatus.SWING_ALL:
            self._contact = np.zeros(4, dtype=int)
            self._phase = np.full(4, 0.5)

        elif status == WaveStatus.STANCE_ALL:
            self._contact = np.ones(4, dtype=int)
            self._phase = np.full(4, 0.5)
