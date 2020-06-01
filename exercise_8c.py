"""Exercise 8c"""

import pickle
import numpy as np
from simulation import simulation
from simulation_parameters import SimulationParameters
from plot_results_2 import main_2


def exercise_8c(timestep, Rhead_vector, Rtail_vector, experience):
    """Exercise 8c"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=3,  # An example of parameter part of the grid search
            amplitudes = [Rheads, Rtails],
            phase_lag=2*np.pi,
            exercise_8c = True,
        )
        for Rheads in Rhead_vector
        for Rtails in Rtail_vector
    ]

    # Grid search
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


if __name__ == '__main__':
    
    experience_name = 'exercise_8c__'
    
    #0.521 seems to be too much (mvmts are too wide !)
    #USUAL RANGE IS: [0.261, 0.521]
    RheadVector = np.linspace(0.06,0.521,1)
    RtailVector = np.linspace(0.351,0.521,1)
    
    color_map_array_blank = np.zeros((len(RheadVector),len(RheadVector),3))
    
    #parameters.distance_reached = np.zeros((len(RheadVector),len(RtailVector)))

    
    exercise_8c(timestep=1e-2, 
                Rhead_vector = RheadVector, 
                Rtail_vector = RtailVector,
                experience = experience_name)

    main_2(exercise = experience_name,
         simulation = [],
         Rhead_vector = RheadVector,
         Rtail_vector = RtailVector,
         color_map_array = color_map_array_blank)
    
    
    