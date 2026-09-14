import numpy as np


class TraceConditioning:
    def __init__(self, ISI, ITI, num_distractors, seed):
        self.mt = np.random.default_rng(seed)
        self.ISI = ISI
        self.ITI = ITI
        self.num_distractors = num_distractors
        self.current_state = [0.0, 0.0]
        self.requires_reset = True
        self.remaining_steps = 0
        self.remaining_until_US = 0
        self.ISI_length = ISI[0]

    def increase_ISI(self, t):
        self.ISI_length += t
        self.ISI = (self.ISI_length, self.ISI_length)

    def get_state(self):
        return self.current_state

    def step(self):
        if self.remaining_steps == 0:
            return self.reset()
        if self.remaining_until_US == 1:
            self.current_state[1] = 1.0
        else:
            self.current_state[1] = 0.0
        self.current_state[0] = 0.0
        self.remaining_until_US -= 1
        self.remaining_steps -= 1
        return self.current_state

    def reset(self):
        self.remaining_until_US = int(self.mt.integers(self.ISI[0], self.ISI[1] + 1))
        self.remaining_steps = self.remaining_until_US + int(
            self.mt.integers(self.ITI[0], self.ITI[1] + 1))
        self.current_state[0] = 1.0  # CS
        self.current_state[1] = 0.0  # US
        return self.current_state

    def get_US(self):
        return self.current_state[1]

    def get_target(self, gamma):
        if self.remaining_until_US > 0:
            return gamma ** (self.remaining_until_US - 1)
        return 0.0

    def cumulant(self):
        return self.current_state[1]


class TracePatterning:
    def __init__(self, ISI, ITI, num_distractors, seed):
        self.mt = np.random.default_rng(seed)
        self.ISI = ISI
        self.ITI = ITI
        self.num_distractors = num_distractors
        self.pattern_len = 6
        self.current_state = [0.0] * (self.pattern_len + self.num_distractors + 1)
        self.remaining_steps = 0
        self.remaining_until_US = 0
        self.valid = False

        self.valid_patterns = []
        while len(self.valid_patterns) < 10:
            p = self.create_pattern()
            if not any(p == existing for existing in self.valid_patterns):
                self.valid_patterns.append(p)

    def get_state(self):
        return self.current_state

    def create_pattern(self):
        temp_state = [0.0] * 6
        while sum(temp_state) != 3:
            s = int(self.mt.integers(0, 6))
            temp_state[s] = 1.0
        return temp_state

    def step(self):
        for a in range(self.pattern_len + self.num_distractors):
            self.current_state[a] = 0.0
        if self.remaining_steps == 0:
            return self.reset()
        self.set_noise_bits()
        if self.remaining_until_US == 1 and self.valid:
            self.current_state[6] = 1.0
        else:
            self.current_state[6] = 0.0
        self.remaining_until_US -= 1
        self.remaining_steps -= 1
        return self.current_state

    def reset(self):
        self.remaining_until_US = int(self.mt.integers(self.ISI[0], self.ISI[1] + 1))
        self.remaining_steps = self.remaining_until_US + int(
            self.mt.integers(self.ITI[0], self.ITI[1] + 1))
        state_pattern = self.create_pattern()
        for a in range(6):
            self.current_state[a] = state_pattern[a]
        self.valid = any(state_pattern == p for p in self.valid_patterns)
        self.set_noise_bits()
        return self.current_state

    def set_noise_bits(self):
        counter = 1.0
        for temp in range(self.pattern_len + 1,
                          self.pattern_len + 1 + self.num_distractors):
            if self.mt.random() < 1 / (counter * 10):
                self.current_state[temp] = 1.0
            else:
                self.current_state[temp] = 0.0
            counter += 1

    def get_US(self):
        return self.current_state[6]

    def get_target(self, gamma):
        if self.remaining_until_US > 0 and self.valid:
            return gamma ** (self.remaining_until_US - 1)
        return 0.0

    def cumulant(self):
        return self.current_state[6]
