"""Exercise 8f"""

import pickle
import numpy as np
import matplotlib.animation as manimation
from simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from plot_results import main


def exercise_8f(timestep,phase_lag_vector,experience):
    """Exercise 8f"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr=2,  # drive so the robot can walk
            amplitude_limb = 0.5,
            phase_lag= phase_lag,  
            phase_limb_body = np.pi/2,
            exercise_8f = True
            # ...
        )
        for phase_lag in phase_lag_vector
    ]

    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/{}/simulation_{}.{}'
        sim, data = simulation( 
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='ground',  # Can also be 'ground' or 'amphibious'
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
    
    experience_name = 'exercise_8f__'
    nb_phase = 2
    nb_simulation = nb_phase
    # phase_lag_vector = np.linspace(0, 3/2 * np.pi, nb_phase)
    phase_lag_vector=  np.array([np.pi/2])     
    exercise_8f(timestep=1e-2, 
                phase_lag_vector = phase_lag_vector,  
                experience = experience_name)
    
    # plot_all_trajectories(phase_lag_vector, 
    #                       amplitude_vector, 
    #                       nb_phase, 
    #                       experience_name,
    #                       color_map_array_blank)
    # main(plot=False, 
    #     exercise = experience_name, 
    #     simulation = simulation,
    #     amplitude_vector = amplitude_vector,
    #     phase_lag_vector = phase_lag_vector,
    #     color_map_array = color_map_array_blank)
    