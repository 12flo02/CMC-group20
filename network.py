"""Oscillator network ODE"""

import numpy as np

from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters

    Return
    ------
    :<np.array>
        Returns derivative of state (phases and amplitudes)
    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    
    # Implement equation here
    phases_dot = 2 * np.pi * robot_parameters.freqs
    
    # frequency = diff(robot_parameters.phase_bias)
    amplitudes_dot = np.zeros_like(amplitudes)


    
    for i in range(0, n_oscillators):
        amplitudes_dot[i] = robot_parameters.rates[i] * \
                            (robot_parameters.nominal_amplitudes[i] - amplitudes[i])      
        
        for j in range(0, n_oscillators):
            phases_dot[i] += amplitudes[j] * robot_parameters.coupling_weights[j][i] * \
                             np.sin(phases[j] - phases[i] - robot_parameters.phase_bias[j][i])

                                 
        
    return np.concatenate([phases_dot, amplitudes_dot])


def motor_output(phases, amplitudes, iteration=None):
    """Motor output.

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.
    """
    # Implement equation here
    alpha = np.linspace(0.5, 1, 10)
    """ rest postion of the limb"""
    beta = np.pi/2
    
    output_motor_body_left = amplitudes[:10] * (1 + np.cos(phases[:10])) 
    output_motor_body_right = amplitudes[10:20] * (1 + np.cos(phases[10:20]))
    output_motor_body = (output_motor_body_left - output_motor_body_right)
    """ if we want to integrate amplitude gradiant like in the paper"""
    # output_motor_body = alpha * (output_motor_body_left - output_motor_body_right)
     
    """ add 90° to has the limb along the body"""
    output_motor_limb = phases[20:24] + beta
    
    
    return np.concatenate([output_motor_body, output_motor_limb])


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls, n_iterations):
        """State of Salamandra robotica 2"""
        shape = (n_iterations, 2*24)
        return cls(
            shape,
            dtype=np.float64,
            buffer=np.zeros(shape)
        )

    def phases(self, iteration=None):
        """Oscillator phases"""
        return self[iteration, :24] if iteration is not None else self[:, :24]

    def set_phases(self, iteration, value):
        """Set phases"""
        self[iteration, :24] = value

    def set_phases_left(self, iteration, value):
        """Set body phases on left side"""
        self[iteration, :10] = value

    def set_phases_right(self, iteration, value):
        """Set body phases on right side"""
        self[iteration, 10:20] = value

    def set_phases_legs(self, iteration, value):
        """Set leg phases"""
        self[iteration, 20:24] = value

    def amplitudes(self, iteration=None):
        """Oscillator amplitudes"""
        return self[iteration, 24:] if iteration is not None else self[:, 24:]

    def set_amplitudes(self, iteration, value):
        """Set amplitudes"""
        self[iteration, 24:] = value


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations):
        super(SalamandraNetwork, self).__init__()
        # States
        self.state = RobotState.salamandra_robotica_2(n_iterations)
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here

# =============================================================================
#         self.state.set_phases(
#             iteration=0,
#             value=np.zeros(self.robot_parameters.n_oscillators),
#         )
# =============================================================================
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.ranf(self.robot_parameters.n_oscillators),
        )
        
# =============================================================================
#         value = 1e-2*np.random.ranf(self.robot_parameters.n_oscillators)
#         self.robot_parameters.freqs = value
# =============================================================================
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

    def step(self, iteration, time, timestep):
        """Step"""
        self.solver.set_f_params(self.robot_parameters)
        self.state[iteration+1, :] = self.solver.integrate(time+timestep)

    def get_outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        phases = self.state.phases(iteration=iteration)
        amplitudes = self.state.amplitudes(iteration=iteration)
        
        output_body_left = amplitudes[:10] * (1 + np.cos(phases[:10])) 
        output_body_right = amplitudes[10:20] * (1 + np.cos(phases[10:20]))
        # output_body = output_body_left - output_body_right
        output_body = output_body_left
     
        output_limb = amplitudes[20:24] * (1 + np.cos(phases[20:24]))
        # output_limb = amplitudes[20:24] * phases[20:24]
        
        return np.concatenate([output_body, output_limb])
    
    
    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )

