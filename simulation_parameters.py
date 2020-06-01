"""Simulation parameters"""
import numpy as np


class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.duration = 30
        self.phase_lag = None
        self.phase_limb_body = None
        self.amplitude_body = None
        self.amplitude_limb = None
        self.amplitude_gradient = None
        self.turn = None
        self.amplitudes = np.zeros(2) #=[Rhead,Rtail]
        self.drive_right = 0
        self.drive_left = 0
        self.current_iter = 0
        self.epsilon = 1e-1
        
        self.best_params = False #to combine and see how the best parameters do
        self.exercise_8b = False
        self.exercise_8c = False
        self.exercise_8d1 = False
        self.exercise_8d2 = False
        self.exercise_8f = False
        self.exercise_example = False
        # Feel free to add more parameters (ex: MLR drive)
        self.drive_mlr = kwargs.get("drive_mlr", 2)
        self.u_turn_params = np.zeros(2)
# =============================================================================
#         self.amplitude_body = kwargs.get("amplitude_doby", 0.326)
#         self.phase_lag = kwargs.get("phase_lag", np.pi/4)
# =============================================================================

        
        # ...
        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)

