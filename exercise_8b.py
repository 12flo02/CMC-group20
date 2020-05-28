"""Exercise 8b"""

import pickle
import numpy as np
import matplotlib.animation as manimation
from simulation import simulation
from simulation_parameters import SimulationParameters
import farms_pylog as pylog
from plot_results import main
from plot_results_2 import main_2

# from tqdm import tqdm

#%%
def exercise_8b(timestep, phase_lag_vector, amplitude_vector, experience):
    """Exercise 8b"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2,  # An example of parameter part of the grid search
            amplitude_body=amplitudes,  # Just an example
            phase_lag=phase_lag,  # or np.zeros(n_joints) for example
            turn=0,
            exercise_8b = True  # Another example
            # ...
        )
        for amplitudes in amplitude_vector
        for phase_lag in phase_lag_vector
        
        # for ...
    ]

    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        print("simulation : %d" %(simulation_i))
        filename = './logs/{}/simulation_{}.{}'
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

#%%
def plot_all_trajectories(phase_lag_vector, amplitude_vector, nb_phase, experience_name):
#%%   

    legend = []
    string = "phase_lag : "
    
    color_map_array_blank = np.zeros((len(amplitude_vector),len(phase_lag_vector),3))
    
    for i in range(nb_phase):
        legend.append("%.2f" %(phase_lag_vector[i]))
    
    # legend = [string + x for x in legend]
    
    for i, amp in enumerate(amplitude_vector) :
        print(i*nb_phase)
        print((i+1)*nb_phase - 1)
        simulation = np.arange(i*nb_phase, (i+1)*nb_phase)   
        print(simulation)
        main(plot=True, 
             exercise = experience_name, 
             simulation = simulation, 
             amplitude = amp, 
             phase_legend = legend)

    main_2(plot=True, 
         exercise = experience_name, 
         simulation = simulation, 
         phase_lag_vector = phase_lag_vector, 
         amplitude_vector=amplitude_vector, 
         color_map_array = color_map_array_blank)
              

#%%   

if __name__ == '__main__':
    
    experience_name = 'exercise_8b__test'
    nb_phase = 3
    nb_amp = 3
    nb_simulation = nb_phase * nb_amp
    
    phase_lag_vector = np.linspace(np.pi, 3/2* np.pi, nb_phase)
    # phase_lag_vector = [3/2*np.pi]
    if nb_amp == 1:
        amplitude_vector = [0.261]
    else:
        amplitude_vector = np.linspace(0.1, 0.2, nb_amp)

    
    exercise_8b(timestep=1e-2, 
                phase_lag_vector = phase_lag_vector, 
                amplitude_vector = amplitude_vector,
                experience = experience_name)
    
    plot_all_trajectories(phase_lag_vector, 
                          amplitude_vector, 
                          nb_phase, 
                          experience_name)
    

