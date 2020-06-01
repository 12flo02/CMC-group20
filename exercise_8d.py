"""Exercise 8d"""

import pickle
import numpy as np
from simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results_2 import main_2


def exercise_8d1(timestep, experience = None, u_turn_params=None):
    """Exercise 8d1"""
    # Use exercise_example.py for reference

    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=3,  # An example of parameter part of the grid search
            amplitudes=[0.351,0.06],  # Just an example
            phase_lag=2*np.pi, # or np.zeros(n_joints) for example
            drive_right = 1,
            drive_left = 4.5,
            exercise_8d1 = True,
            u_turn_params = u_turn_params
        )
    ]

    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/{}/simulation_{}.{}'
        sim, data = simulation( 
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground' or 'amphibious'
            # fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
            # video_distance=1.5,  # Set distance of camera to robot
            # video_yaw=0,  # Set camera yaw for recording
            # video_pitch=-45,  # Set camera pitch for recording
        )
        # Log robot data
        data.to_file(filename.format(experience, simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(experience, simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
        # Save video
        if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))


def exercise_8d2(timestep):
    """Exercise 8d2"""
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=3,  # An example of parameter part of the grid search
            amplitudes=0,  # Just an example
            phase_lag=0,  # or np.zeros(n_joints) for example
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
        # Log robot data
        data.to_file(filename.format(experience, simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(experience, simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
        # Save video
        if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))


if __name__ == '__main__':
    """ FIRST PART  ---   EXERCISE 8D1   --- """
    experience_name = 'exercise_8d1__' 
    u_turn_drive_change = [500,580]
    
    exercise_8d1(timestep=1e-2,
                 experience = experience_name,
                 u_turn_params=u_turn_drive_change)
    
    main_2(plot=False, 
         exercise = experience_name, 
         simulation = [],
         u_turn_params=u_turn_drive_change)
    

    """ FIRST PART  ---   EXERCISE 8D2   --- """
    #exercise_8d2(timestep=1e-2)

