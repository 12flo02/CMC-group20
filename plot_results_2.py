"""Plot results"""

import pickle
import math
import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from save_figures import save_figures
from parse_args import save_plots
from salamandra_simulation.data import AnimatData
from simulation import simulation
from simulation_parameters import SimulationParameters



def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    
    
def plot_specific_plot(data_array, x_data, y_data, legend, exercise, plot_name, plot_title):
    
    plt.figure(str(plot_name) + str(exercise), 
               figsize=[8, 6],
               dpi = 300)
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data_array, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel(legend[0])
    plt.ylabel(legend[1])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(len(x_data)), x_data)
    plt.yticks(np.arange(len(y_data)), y_data)
    plt.title('Grid Search with ' + str(plot_title) + ' traveled Score')
    plt.savefig("graphs/" + str(plot_name) + "_" + str(exercise) + ".png")
    # plt.savefig("graphs/distance_" + str(exercise) + ".eps")
    plt.show()

    return

def plot_color_plots(exercise = None, data_array = None, simulation = None, phase_lag_vector = None, amplitude_vector=None, Rhead_vector=None, Rtail_vector=None, data_array_full = False):
    
    plot_name = ['distance', 'energy', 'speed']
    plot_title = ['distance traveled', 'energy', 'speed']
    
    if exercise == 'exercise_8b__':
        x_data = [round(num, 3) for num in phase_lag_vector]
        y_data = [round(num, 3) for num in amplitude_vector]
        legend = ['phase lag vector', 'amplitude vector']
    elif exercise == 'exercise_8c__':
        x_data = [round(num, 3) for num in Rhead_vector]
        y_data = [round(num, 3) for num in Rtail_vector]
        legend = ['head amplitudes', 'tail amplitudes']
    
    """ PLOT :
        - DISTANCE
        - ENERGY
        - SPEED """
    for i in range(0, 3) :
        plot_specific_plot(data_array[:,:,i], x_data, y_data, legend, exercise, plot_name[i], plot_title[i])
        
        
    return


def main_2(plot=True, exercise = None, simulation = [], phase_lag_vector = None, amplitude_vector=None, Rhead_vector=None, Rtail_vector=None, color_map_array = None):
    
    """Main"""
    # Load data    
    if exercise == None:
        data = AnimatData.from_file('logs/example/simulation_0.h5', 2*14) 
        with open('./logs/example/simulation_0.pickle', 'rb') as param_file:
            parameters = pickle.load(param_file)
        times = data.times
        timestep = times[1] - times[0]  # Or parameters.timestep
# =============================================================================
#             amplitudes = parameters.amplitudes            
# =============================================================================
        #amplitudes = parameters.amplitude_body
        phase_lag = parameters.phase_lag
        osc_phases = data.state.phases_all()
        osc_amplitudes = data.state.amplitudes_all()
        links_positions = data.sensors.gps.urdf_positions()
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 10, :]
        joints_positions = data.sensors.proprioception.positions_all()
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()
        # Notes:
        # For the gps arrays: positions[iteration, link_id, xyz]
        # For the positions arrays: positions[iteration, xyz]
        # For the joints arrays: positions[iteration, joint]
        
        
    """ ==================================================================="""    
    if exercise == 'exercise_8b__' :
        simulation = range(len(amplitude_vector)*len(phase_lag_vector))
        for i, simul in enumerate(simulation):
            filename = './logs/{}/simulation_{}.h5'
            filename = filename.format(exercise, simul)
            data = AnimatData.from_file(filename, 2*14)  
            print(filename)
            
            filename = './logs/{}/simulation_{}.pickle'
            filename = filename.format(exercise, simul)

            with open(filename, 'rb') as param_file:
                parameters = pickle.load(param_file)
            times = data.times
            timestep = times[1] - times[0]  # Or parameters.timestep
            osc_phases = np.asarray(data.state.phases_all())
            osc_amplitudes = np.asarray(data.state.amplitudes_all())
            links_positions = np.asarray(data.sensors.gps.urdf_positions())
            head_positions = np.asarray(links_positions[:, 0, :])
            tail_positions = np.asarray(links_positions[:, 10, :])
            joints_positions = np.asarray(data.sensors.proprioception.positions_all())
            joints_velocities = np.asarray(data.sensors.proprioception.velocities_all())
            joints_torques = np.asarray(data.sensors.proprioception.motor_torques())
            
            
            energy = np.absolute(joints_velocities[:,1:11]*joints_torques[:,1:11])
            energy = np.sum(energy*timestep)
            
            head_dist = (head_positions[-1,0]**2+head_positions[-1,1]**2)**0.5
            tail_dist = (tail_positions[-1,0]**2+tail_positions[-1,1]**2)**0.5
            
            if(head_dist < tail_dist):
                traveled_dist = -head_dist
            else:
                traveled_dist = head_dist
                
            
            head_speed = np.zeros((1000,3))
            for j in range(999):
                head_speed[i+1,:] = (head_positions[i+1,:]-head_positions[i,:])/timestep
            head_tot_speed = max((np.sum(head_speed**2,axis=1))**0.5)
            
            #This just makes sure the indexes are correct. Simpler solutions may exist...
            length = len(phase_lag_vector)
            if (i==0):
                k=0
                l=0
            else:
                k = int(math.floor(i/len(phase_lag_vector)))
                l = i%length
                
            color_map_array[k,l,0] = traveled_dist
            color_map_array[k,l,1] = energy
            color_map_array[k,l,2] = head_tot_speed
            if(k == len(phase_lag_vector)-1 and l == len(phase_lag_vector)-1):
                plot_color_plots(exercise = exercise,
                                 data_array = color_map_array,
                                 simulation = simulation,
                                 phase_lag_vector = phase_lag_vector,
                                 amplitude_vector=amplitude_vector,
                                 Rhead_vector=Rhead_vector,
                                 Rtail_vector=Rtail_vector)
        
        """DISCUTER AVEC FLO SI/COMMENT REINTEGRER CA PROPREMENT"""    
        #     plt.figure("Trajectory_amplitude_%.3f" %(amplitude))
        #     plt.title("Trajectory_amplitude_%.3f" %(amplitude))
        #     plot_trajectory(head_positions)
        
        # if phase_legend != None :
        #     string ="phase lag : "
        #     phase_legend = [ string + x for x in phase_legend]
        #     plt.legend(phase_legend, loc = 2, fontsize = "small")
        """DISCUTER AVEC FLO SI/COMMENT REINTEGRER CA PROPREMENT"""       
            
        
    """ ==================================================================="""        
    if exercise == 'exercise_8c__':
        simulation = range(len(Rhead_vector)*len(Rtail_vector))
        for i, simul in enumerate(simulation):
            filename = './logs/{}/simulation_{}.h5'
            filename = filename.format(exercise, simul)
            data = AnimatData.from_file(filename, 2*14)  
            print(filename)
            
            filename = './logs/{}/simulation_{}.pickle'
            filename = filename.format(exercise, simul)

            with open(filename, 'rb') as param_file:
                parameters = pickle.load(param_file)
            times = data.times
            timestep = times[1] - times[0]  # Or parameters.timestep
            
            
# =============================================================================
#           amplitudes = parameters.amplitudes            
# =============================================================================
            amplitudes = parameters.amplitude_body
            phase_lag = parameters.phase_lag
            osc_phases = np.asarray(data.state.phases_all())
            osc_amplitudes = np.asarray(data.state.amplitudes_all())
            links_positions = np.asarray(data.sensors.gps.urdf_positions())
            head_positions = np.asarray(links_positions[:, 0, :])
            tail_positions = np.asarray(links_positions[:, 10, :])
            joints_positions = np.asarray(data.sensors.proprioception.positions_all())
            joints_velocities = np.asarray(data.sensors.proprioception.velocities_all())
            joints_torques = np.asarray(data.sensors.proprioception.motor_torques())
            
            head_dist = (head_positions[-1,0]**2+head_positions[-1,1]**2)**0.5
            tail_dist = (tail_positions[-1,0]**2+tail_positions[-1,1]**2)**0.5
            
            if(head_dist < tail_dist):
                traveled_dist = -head_dist
            else:
                traveled_dist = head_dist
            
            energy = np.absolute(joints_velocities[:,1:11]*joints_torques[:,1:11])
            energy = np.sum(energy*timestep)
            
            
            # traveled_dist = (np.sum((head_positions[-1,:]+[0,0,0.1])**2))**0.5 #to compensate spawn at (0,0,0.1)
            # if (head_positions[-1,0] < -0.1 and head_positions[-1,2] < -0.1):
            #     traveled_dist = -traveled_dist
            
            head_speed = np.zeros((1000,3))
            for j in range(999):
                head_speed[i+1,:] = (head_positions[i+1,:]-head_positions[i,:])/timestep
            head_tot_speed = max((np.sum(head_speed**2,axis=1))**0.5)
            
            #This just makes sure the indexes are correct. Simpler solutions may exist...
            length = len(Rhead_vector)
            if (i==0):
                k=0
                l=0
            else:
                k = int(math.floor(i/len(Rhead_vector)))
                l = i%length
                
            color_map_array[k,l,0] = traveled_dist
            color_map_array[k,l,1] = energy
            color_map_array[k,l,2] = head_tot_speed
            if(k == len(Rhead_vector)-1 and l == len(Rhead_vector)-1):
                plot_color_plots(exercise = exercise, 
                                 data_array = color_map_array, 
                                 simulation = simulation, 
                                 phase_lag_vector = phase_lag_vector, 
                                 amplitude_vector=amplitude_vector, 
                                 Rhead_vector=Rhead_vector, 
                                 Rtail_vector=Rtail_vector)
        
            
    """ ==================================================================="""
    if exercise == 'exercise_8d1__':
        simulation = range(len(phase_lag_vector))
        for i, simul in enumerate(simulation):
            filename = './logs/{}/simulation_{}.h5'
            filename = filename.format(exercise, simul)
            data = AnimatData.from_file(filename, 2*14)  
            print(filename)
            
            filename = './logs/{}/simulation_{}.pickle'
            filename = filename.format(exercise, simul)

            with open(filename, 'rb') as param_file:
                parameters = pickle.load(param_file)
            times = data.times
            timestep = times[1] - times[0]  # Or parameters.timestep
            osc_phases = np.asarray(data.state.phases_all())
            osc_amplitudes = np.asarray(data.state.amplitudes_all())
            links_positions = np.asarray(data.sensors.gps.urdf_positions())
            head_positions = np.asarray(links_positions[:, 0, :])
            tail_positions = np.asarray(links_positions[:, 10, :])
            joints_positions = np.asarray(data.sensors.proprioception.positions_all())
            joints_velocities = np.asarray(data.sensors.proprioception.velocities_all())
            joints_torques = np.asarray(data.sensors.proprioception.motor_torques())
            
            plt.figure("Total phase lag of " + str(phase_lag_vector[i]))
            plt.title("Total phase lag of " + str(phase_lag_vector[i]))
            plot_trajectory(head_positions)
    
    """ ==================================================================="""
    if exercise == 'exercise_8d2__':
        simulation = range(1)
        for i, simul in enumerate(simulation):
            filename = './logs/{}/simulation_{}.h5'
            filename = filename.format(exercise, simul)
            data = AnimatData.from_file(filename, 2*14)  
            print(filename)
            
            filename = './logs/{}/simulation_{}.pickle'
            filename = filename.format(exercise, simul)

            with open(filename, 'rb') as param_file:
                parameters = pickle.load(param_file)
            times = data.times
            timestep = times[1] - times[0]  # Or parameters.timestep
            osc_phases = np.asarray(data.state.phases_all())
            osc_amplitudes = np.asarray(data.state.amplitudes_all())
            links_positions = np.asarray(data.sensors.gps.urdf_positions())
            head_positions = np.asarray(links_positions[:, 0, :])
            tail_positions = np.asarray(links_positions[:, 10, :])
            joints_positions = np.asarray(data.sensors.proprioception.positions_all())
            joints_velocities = np.asarray(data.sensors.proprioception.velocities_all())
            joints_torques = np.asarray(data.sensors.proprioception.motor_torques())
            
            plt.figure("Exercise 8d2 - salamandra moves backwards")
            plt.title("Exercise 8d2 - salamandra moves backwards")
            plot_trajectory(head_positions)
        
    
        
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
        
    
    

if __name__ == '__main__':
    main_2(plot=not save_plots())

