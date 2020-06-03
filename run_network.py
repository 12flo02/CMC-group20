"""Run network without Pybullet"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import farms_pylog as pylog
from network import SalamandraNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters
from tqdm import tqdm

#%%
def run_network(duration, update=True, drive=0):
#

    """Run network without Pybullet and plot results
    Parameters
    ----------
    duration: <float>
        Duration in [s] for which the network should be run
    update: <bool>
        description
    drive: <float/array>
        Central drive to the oscillators
    """
    # Simulation setup
    # timestep = 1e-2
    timestep = 1e-1
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    sim_parameters = SimulationParameters(
        drive_mlr=drive,
        amplitude_gradient=None,
        phase_lag=None,
        turn=None,
# =============================================================================
#         drive_mlr=2,  # drive so the robot can walk
#         amplitude_limb = 5,
#         phase_lag= np.pi/2, 
#         phase_limb_body = np.pi/2,
#         exercise_8f = True
# =============================================================================
        # ...
    )
    network = SalamandraNetwork(sim_parameters, n_iterations)
    osc_left = np.arange(10)
    osc_right = np.arange(10, 20)
    osc_legs = np.arange(20, 24)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases(iteration=0))
    ])
    phases_log[0, :] = network.state.phases(iteration=0)
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes(iteration=0))
    ])
    amplitudes_log[0, :] = network.state.amplitudes(iteration=0)
    freqs_log = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_log[0, :] = network.robot_parameters.freqs
    
    freqs_calculated = np.zeros([
        n_iterations,
        len(network.robot_parameters.freqs)
    ])
    freqs_calculated[0, :] = network.robot_parameters.freqs
    
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output(iteration=0))
    ])
    outputs_log[0, :] = network.get_motor_position_output(iteration=0)
    
    drive = np.linspace(0.5, 5.5, len(times))

    # Run network ODE and log data
    tic = time.time()
    for i, time0 in enumerate(tqdm(times[1:])):
    # for i, time0 in enumerate(times[1:]):
        if update:
            network.robot_parameters.update(
                SimulationParameters(
                    drive_mlr = drive[i+1],
                    # amplitude_gradient=None,
                    # phase_lag = 1.5
                ),
            )
        network.step(i, time0, timestep)
        phases_log[i+1, :] = network.state.phases(iteration=i+1)
        amplitudes_log[i+1, :] = network.state.amplitudes(iteration=i+1)
        # outputs_log[i+1, :] = network.get_motor_position_output(iteration=i+1)
        """ CHANGE THE FUNCTION FOR OUTPUTS"""
        outputs_log[i+1, :] = network.get_outputs(iteration=i+1)
        freqs_log[i+1, :] = network.robot_parameters.freqs        
        # freqs_calculated = 
    # # Alternative option
    # phases_log[:, :] = network.state.phases()
    # amplitudes_log[:, :] = network.state.amplitudes()
    # outputs_log[:, :] = network.get_motor_position_output()
    toc = time.time()
    
    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))
    
    return times, drive, freqs_log, amplitudes_log, outputs_log, phases_log



#%%
def find_start_stop_salamandra(freqs_log):
    
    start = np.where(freqs_log[1:,0] > 0.2)[0][0]
    stop_limb = np.argmax(freqs_log[1:,0]) + 1
    stop_body = np.argmax(freqs_log[1:,20]) + 1
    
    return np.array([start, stop_limb,stop_body])

#%%
def plot_all_graphs(times, drive, freqs_log, amplitudes_log, outputs_log, phases_log):

    # Implement plots of network results
    timestep = 1e-1
    index = timestep*find_start_stop_salamandra(freqs_log)
    
    output_body_label = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$",
                        "$x_5$", "$x_6$", "$x_7$", "$x_8$", "$x_9$"]
# =============================================================================
#     
#     plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
#            axisbelow=True, grid=True)
#     plt.rc('grid', color='w', linestyle='solid')
# =============================================================================
    
    """First figure, amplitudes and frequency as a function of the drive"""    
    fig1 = plt.figure("frequency and Amplitude", 
                      figsize = [8,6],
                      dpi = 600,
                      constrained_layout=True)
# =============================================================================
#     plt.grid(color='w', linestyle='solid')
# =============================================================================
    
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1)
    
    ax_freq = fig1.add_subplot(gs[0,:])
    ax_amp = fig1.add_subplot(gs[1,:])
    
    
    ax_freq.plot(drive[1:], freqs_log[1:,0], c='k')
    ax_freq.plot(drive[1:], freqs_log[1:,20], c='k', ls='--')
    ax_freq.set_ylabel("freqeuncy [Hz]")
    ax_freq.legend(["body", "limb"], loc=2)
    ax_freq.set_xlim(min(drive), max(drive))
    ax_freq.set_ylim(0, 1.5)
    
    ax_amp.plot(drive[1:], amplitudes_log[1:,0], c='k')
    ax_amp.plot(drive[1:], amplitudes_log[1:,20], c='k', ls='--')
    ax_amp.set_xlabel("drive")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.legend(["body", "limb"], loc=2)
    ax_amp.set_xlim(min(drive), max(drive))
    ax_amp.set_ylim(0, 0.7)
     
    plt.savefig("Figure_3b_Ijspeert.jpg")
    plt.savefig("Figure_3b_Ijspeert.eps")
    
    
    """ Second  figure, drive,, frequency, limb output, body output as a function of time"""
    fig2 = plt.figure("motor output", 
                      figsize = [8,6],
                      dpi = 600,
                      constrained_layout=True)
    
    gs = gridspec.GridSpec(ncols=1, nrows=4, figure=fig2)
    
    ax_x_body = fig2.add_subplot(gs[0,:])
    ax_x_limb = fig2.add_subplot(gs[1,:])
    ax_freqs = fig2.add_subplot(gs[2,:])
    ax_drive = fig2.add_subplot(gs[3,:])
    
    
    """X BODY"""
    """head blue for trunk ans green for tail"""
    for _, idx in enumerate(index):
        ax_x_body.axvline(x=idx, ymin = -50, ymax = 50, c="0.8", linewidth=1.0, ls=":")
    for i in range(1,5):
        ax_x_body.plot(times[1:], outputs_log[1:,i] - 2*i, c = 'b')
    for i in range(5,10):
        ax_x_body.plot(times[1:], outputs_log[1:,i] - 2*i, c = 'g')
        
    ax_x_body.grid(color='w', linestyle='solid')
    
    ax_x_body.set_ylabel("x Body")
    ax_x_body.set_xticks([])
    ax_x_body.set_yticks([])
    ax_x_body.set_xlim(0,40)
    # ax_x_body.set_ylim(-20,20)
    ax_x_body.text(1.0, -4.5, "trunk", size=10, rotation=90.,
                   ha="center", va="center",
                   bbox=dict(boxstyle="round",ec='b',fc='w',))
    ax_x_body.text(1.0, -14, "tail", size=10, rotation=90.,
                   ha="center", va="center",
                   bbox=dict(boxstyle="round",ec='g',fc='w',))
    
    
    for i in range (1, 10):
        ax_x_body.text(3.0, 0.9 - 2*i, "$x_{}$".format(i), size=6, rotation=0.,
                       ha="center", va="center")

    
    
    
    """X LIMB"""
    for _, idx in enumerate(index):
        ax_x_limb.axvline(x=idx, ymin = -50, ymax = 50, c="0.8", linewidth=1.0, ls=":")  
    ax_x_limb.plot(times[1:], outputs_log[1:,10], c = 'b')
    ax_x_limb.plot(times[1:], outputs_log[1:,11] - 1.1, c = 'b')
    ax_x_limb.plot(times[1:], outputs_log[1:,12] - 2.2, c = 'g')  
    ax_x_limb.plot(times[1:], outputs_log[1:,13] - 3.3, c = 'g')  

        
    ax_x_limb.set_ylabel("x Limb")
    ax_x_limb.set_xticks([])
    ax_x_limb.set_yticks([])
    ax_x_limb.set_xlim(0,40)
    # ax_x_limb.set_ylim(-1.1,1.1)
    ax_x_limb.text(1.0, -0.5, "trunk", size=10, rotation=90.,
                   ha="center", va="center",
                   bbox=dict(boxstyle="round",ec='b',fc='w',))
    ax_x_limb.text(1.0, -2.5, "tail", size=10, rotation=90.,
                   ha="center", va="center",
                   bbox=dict(boxstyle="round",ec='g',fc='w',))
    
    for i in range (20,24):
        label = '$x_{\mathrm{%s}}$' % (i)
        ax_x_limb.text(3.0, 0.3 - 1.1*(i-20), label, size=6, rotation=0.,
                       ha="center", va="center")
        


    """FREQUENCY"""
    instant_freq = np.diff(phases_log, n=1, axis = 0)/(2*np.pi*timestep) 
    for _, idx in enumerate(index):        
        ax_freqs.axvline(x=idx, ymin = -50, ymax = 50, c="0.8", linewidth=1.0, ls=":") 
        for i in range(0,24 ):
            ax_freqs.plot(times[1:], instant_freq[:,i], c= 'k')
         
    ax_freqs.set_ylabel("freqeuncy [Hz]")
    ax_freqs.set_xticks([])
    ax_freqs.set_xlim(0,40)
    ax_freqs.set_ylim(0, 1.5)
    
    
    """DRIVE"""
    for _, idx in enumerate(index):
        ax_drive.axvline(x=idx, ymin = -50, ymax = 50, c="0.8", linewidth=1.0, ls=":") 
        
    ax_drive.plot(times[1:], drive[1:], c = 'k')
    ax_drive.axhline(y=1, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0)  
    ax_drive.axhline(y=5, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0) 
    ax_drive.axhline(y=3, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0) 
   
    ax_drive.set_xlim(0,40)
    ax_drive.set_ylim(0,6)
    ax_drive.set_xlabel("time [s]")
    ax_drive.set_ylabel("drive")
    
    ax_drive.text(4.5, 2, "walking", size=10, rotation=0.,)
    ax_drive.text(20.5, 4, "swimming", size=10, rotation=0.,)
    
    plt.savefig("Figure_3a_Ijspeert.jpg")
    plt.savefig("Figure_3a_Ijspeert.eps")
    plt.show()
    
    
    
    
    """ Third  figure, drive,, frequency, limb output, body output uncoupled"""
    fig3 = plt.figure("uncoupled output", 
                      figsize = [8,6],
                      dpi = 600,
                      constrained_layout=True
                      )
    
    gs = gridspec.GridSpec(ncols=1, nrows=4, figure=fig3)
    
    ax_x_body = fig3.add_subplot(gs[0,:])
    ax_x_freqs = fig3.add_subplot(gs[1,:])
    ax_amp = fig3.add_subplot(gs[2,:])
    ax_drive = fig3.add_subplot(gs[3,:])
    
    
    """X BODY AND LIMB"""
    ax_x_body.plot(times[1:], outputs_log[1:,1], c = 'k')
    ax_x_body.plot(times[1:], outputs_log[1:,10] - 2, c = 'k', ls = "--")  
        
    ax_x_body.set_ylabel("x")
    ax_x_body.set_xticks([])
    ax_x_body.set_yticks([])
    ax_x_body.set_xlim(0,40)


    """FREQUENCY"""
    ax_x_freqs.plot(times[1:], freqs_log[1:,1], c= 'k')
    ax_x_freqs.plot(times[1:], freqs_log[1:,20], c= 'k', ls = "--")
         
    ax_x_freqs.set_ylabel("freqeuncy [Hz]")
    ax_x_freqs.set_xticks([])
    ax_x_freqs.set_xlim(0,40)
    ax_x_freqs.set_ylim(0, 1.5)
    
    
    """AMPLITUDE"""
    ax_amp.plot(times[1:], amplitudes_log[1:,1], c= 'k')
    ax_amp.plot(times[1:], amplitudes_log[1:,20], c= 'k', ls = "--")
         
    ax_amp.set_ylabel("R")
    ax_amp.set_xticks([])
    ax_amp.set_xlim(0,40)
    ax_amp.set_ylim(0, 0.6)
    
    """DRIVE"""      
    ax_drive.plot(times[1:], drive[1:], c = 'k')
    ax_drive.axhline(y=1, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0)  
    ax_drive.axhline(y=5, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0) 
    ax_drive.axhline(y=3, xmin=0, xmax=40, ls = '--', c = '#F6AD1B', linewidth=1.0) 
   
    ax_drive.set_xlim(0,40)
    ax_drive.set_ylim(0,6)
    ax_drive.set_xlabel("time [s]")
    ax_drive.set_ylabel("drive")
    

    
    plt.savefig("Figure_3c_Ijspeert.jpg")
    plt.savefig("Figure_3c_Ijspeert.eps")
    plt.show()
    
    
    
    
    return


#%%

# =============================================================================
# def main(plot):
#     """Main"""
# 
#     run_network(duration=5)
# 
#     # Show plots
#     if plot:
#         plt.show()
#     else:
#         save_figures()
# 
# 
# if __name__ == '__main__':
#     main(plot=not save_plots())
# =============================================================================

times, drive, freqs_log, amplitudes_log, outputs_log, phases_log = run_network(duration=40, update = True, drive = 2)
#%%
plot_all_graphs(times, drive, freqs_log, amplitudes_log, outputs_log, phases_log)



