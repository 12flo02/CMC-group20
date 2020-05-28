"""Exercise 8d"""

import pickle
import numpy as np
from simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results import main


def exercise_8d1(timestep, phase_lag_vector = None, experience = None):
    """Exercise 8d1"""
    # Use exercise_example.py for reference

    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=2,  # An example of parameter part of the grid search
            amplitudes=0,  # Just an example
            phase_lag=phase_lag, # or np.zeros(n_joints) for example
            turn=0,  # Another example
            drive_right = 1,
            drive_left = 4.5,
            # ...
            exercise_8b = False,
            exercise_8c = False,
            exercise_8d1 = True,
            exercise_8d2 = False,
            best_params = False
        )
        for phase_lag in phase_lag_vector
    ]

    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation( 
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
            # video_distance=1.5,  # Set distance of camera to robot
            # video_yaw=0,  # Set camera yaw for recording
            # video_pitch=-45,  # Set camera pitch for recording
        )
    pass


def exercise_8d2(timestep):
    """Exercise 8d2"""
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=2,  # An example of parameter part of the grid search
            amplitudes=0,  # Just an example
            phase_lag=0,  # or np.zeros(n_joints) for example
            turn=0,  # Another example
            drive_right = 1,
            drive_left = 4.5,
            # ...
            exercise_8b = False,
            exercise_8c = False,
            exercise_8d1 = False,
            exercise_8d2 = True
        )
    ]

    for simulation_i, sim_parameters in enumerate(parameter_set):
        sim, data = simulation( 
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            #fast=True,  # For fast mode (not real-time)
            #headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
            # video_distance=1.5,  # Set distance of camera to robot
            # video_yaw=0,  # Set camera yaw for recording
            # video_pitch=-45,  # Set camera pitch for recording
        )


    pass


if __name__ == '__main__':
    """ FIRST PART  ---   EXERCISE 8D1   --- """
    experience_name = 'exercise_8d__'
    nb_phase = 2
    nb_simulation = nb_phase
    
    phase_lag_vector = np.linspace(0, 2*np.pi, nb_phase)
    
    exercise_8d1(timestep=1e-2, 
                 phase_lag_vector = phase_lag_vector,
                 experience = experience_name)
    
    main(plot=False, 
         exercise = experience_name, 
         simulation = [],
         phase_lag_vector = phase_lag_vector)
    

    """ FIRST PART  ---   EXERCISE 8D2   --- """
    #exercise_8d2(timestep=1e-2)

