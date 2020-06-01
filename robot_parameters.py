"""Robot parameters"""

import numpy as np
import farms_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)
        

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # phi_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

        

    def set_frequencies(self, parameters):
        """Set frequencies"""
        # pylog.warning("Coupling weights must be set")
        
        """ drive_body[0] = d_low
            drive_body[1] = d_high """
        drive_body = np.array([1.0, 5.0])
        drive_limb = np.array([1.0, 3.0])
        
        """ c_nu[0] = slope
            c_nu[1] = offset """
        c_nu_body = np.array([0.2, 0.3])
        c_nu_limb = np.array([0.2, 0.0])
        nu_body_sat = 0
        nu_limb_sat = 0
                
        freqs_gradient = np.linspace(-0.1, 0.1, 10)
        
        #################  EXERCISE 8B  #################
        if parameters.exercise_8b:
            if drive_body[0] <= parameters.drive_mlr <= drive_body[1]:
                self.freqs[0:10] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
                self.freqs[10:20] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
            else:
                self.freqs[0:20] = nu_body_sat
        #################  EXERCISE 8C  #################
        elif parameters.exercise_8c:
            self.freqs[0:20] = 1
        #################  EXERCISE 8D1  #################
        elif parameters.exercise_8d1:
            if drive_body[0] <= parameters.drive_mlr <= drive_body[1]:
                self.freqs[0:10] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
                self.freqs[10:20] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
                self.freqs[20:24] = nu_limb_sat 
            else:
                self.freqs[0:20] = nu_body_sat
        ################# EXERCISE 8F  #################
        elif parameters.exercise_8f:
            """ all the oscillators have the same frequency if 1 < drive < 3 """
            if drive_limb[0] <= parameters.drive_mlr < drive_limb[1]:
                self.freqs[0:24] = c_nu_limb[0] * parameters.drive_mlr + c_nu_limb[1]
                
            elif drive_limb[1] <= parameters.drive_mlr <= drive_body[1]:
                self.freqs[0:20] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1]
                self.freqs[20:24] = nu_limb_sat
            else:
                self.freqs[0:20] = nu_body_sat
                self.freqs[20:24] = nu_limb_sat 
        else:  
            """ uncomment this part to have the correct graph for the frequency as a function of the drive"""
            if drive_body[0] <= parameters.drive_mlr <= drive_body[1]:
                self.freqs[0:10] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
                self.freqs[10:20] = c_nu_body[0] * parameters.drive_mlr + c_nu_body[1] + freqs_gradient
            else:
                self.freqs[0:20] = nu_body_sat
                
            if drive_limb[0] <= parameters.drive_mlr <= drive_limb[1]:
                self.freqs[20:24] = c_nu_limb[0] * parameters.drive_mlr + c_nu_limb[1]
            else:
                self.freqs[20:24] = nu_limb_sat
        
     
            
    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        # pylog.warning("Nominal amplitudes must be set")
        
        """ drive_body[0] = d_low
            drive_body[1] = d_high """
        drive_body = np.array([1.0, 5.0])
        drive_limb = np.array([1.0, 3.0])
        
        """ c_amp[0] = slope
            c_amp[1] = offset """
        c_amp_body = np.array([0.065, 0.196])
        c_amp_limb = np.array([0.131, 0.131])
        amp_body_sat = 0
        amp_limb_sat = 0
        
        
        """ Exercise 8b to 8d : Only in water --> don't need to impletment limb amplitudes"""
        if parameters.exercise_8b:
            self.nominal_amplitudes[0:20] = parameters.amplitude_body
            self.nominal_amplitudes[20:] = 0
            """ This code changes params """
        elif parameters.exercise_8c:
            [amp_head, amp_tail] = parameters.amplitudes
            self.nominal_amplitudes[0:10] = np.linspace(amp_head, amp_tail, 10)
            self.nominal_amplitudes[10:20] = np.linspace(amp_head, amp_tail, 10)
            self.nominal_amplitudes[20:] = 0
        elif parameters.exercise_8d1:
            if(parameters.u_turn_params[0] < parameters.current_iter < parameters.u_turn_params[1]):
                self.nominal_amplitudes[0:10] = c_amp_body[0] * parameters.drive_left + c_amp_body[1]
                self.nominal_amplitudes[10:20] = c_amp_body[0] * parameters.drive_right + c_amp_body[1]
                self.nominal_amplitudes[20:] = 0
            else:
                self.nominal_amplitudes[0:10] = np.linspace(parameters.amplitudes[1], parameters.amplitudes[0], 10)
                self.nominal_amplitudes[10:20] = np.linspace(parameters.amplitudes[1], parameters.amplitudes[0], 10)
                self.nominal_amplitudes[20:] = 0
        elif parameters.exercise_example:
            self.nominal_amplitudes[:20] = parameters.amplitude_body
            self.nominal_amplitudes[20:] = parameters.amplitude_limb
        else:
            if drive_body[0] <= parameters.drive_mlr <= drive_body[1]:
                self.nominal_amplitudes[0:20] = c_amp_body[0] * parameters.drive_mlr + c_amp_body[1]
            else:
                self.nominal_amplitudes[0:20] = amp_body_sat
                
            if drive_limb[0] <= parameters.drive_mlr <= drive_limb[1]:
                self.nominal_amplitudes[20:24] = c_amp_limb[0] * parameters.drive_mlr + c_amp_limb[1]
            else:
                self.nominal_amplitudes[20:24] = amp_limb_sat

            

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        # pylog.warning("Coupling weights must be set")
        drive_limb = np.array([1.0, 3.0])
        
        """ Limb Antiphase """
        limb_antiphase = np.array([[20, 21], [21, 20], [20, 22], [22, 20], \
                                  [21, 23], [23, 21], [22, 23], [23, 22]])
                        
        for j in range(0, len(limb_antiphase)) :                   
            self.coupling_weights[limb_antiphase[j][0]][limb_antiphase[j][1]] = 10
            
    
        
        for i in range(1, self.n_oscillators):
            
            """ Axial CPG, travelling wave """
            if i < (self.n_oscillators_body - 1) and i != 9 :
                self.coupling_weights[i][i+1] = 10
                self.coupling_weights[i+1][i] = 10

            
            """ Body Antiphase """
            if i < self.n_oscillators_body/2 :
                self.coupling_weights[i][i+10] = 10
                self.coupling_weights[i+10][i] = 10
                   
        """ In phase """
        for j in range(0,5):
            self.coupling_weights[20][j] = 30
        for j in range(10,15):
            self.coupling_weights[21][j] = 30
        for j in range(5,10):
            self.coupling_weights[22][j] = 30
        for j in range(15,20):
            self.coupling_weights[23][j] = 30


    def set_phase_bias(self, parameters):
        """Set phase bias"""
        # pylog.warning("Phase bias must be set")
        
        drive_limb = np.array([1.0, 3.0])
        
            
            
        """ 11 elements because we dont take the first element phase_lag = 0 """
        """ Exercice 8b """
        if parameters.exercise_8b :
            phase_lag_body_axial = parameters.phase_lag
            
        elif parameters.exercise_8c:
            phase_lag_body_axial = parameters.phase_lag
            
        elif parameters.exercise_8d1:
            phase_lag_body_axial = parameters.phase_lag
            
        elif parameters.exercise_8d2:
            phase_lag_body_axial = -parameters.phase_lag
            
        elif parameters.exercise_8f:
            phase_lag_body_axial = 0
            phase_lag_limb_body = 0
        
        elif parameters.exercise_example:
            phase_lag_body_axial = parameters.phase_lag
        else:
            if parameters.phase_lag:
                phase_lag_body_axial = parameters.phase_lag
            else:
                phase_lag_body_axial = 2*np.pi

        
        """ Limb Antiphase """
        limb_antiphase = np.array([[20, 21], [21, 20], [20, 22], [22, 20], \
                                  [21, 23], [23, 21], [22, 23], [23, 22]])
        
                    
        for j in range(0, len(limb_antiphase)) : 
            self.phase_bias[limb_antiphase[j][0]][limb_antiphase[j][1]] = np.pi
        
        for i in range(1, self.n_oscillators):
            
                
            """ Axial CPG, travelling wave """
            if i < (self.n_oscillators_body - 1) and i != 9 :
                self.phase_bias[i][i+1] = phase_lag_body_axial/10
                self.phase_bias[i+1][i] = -phase_lag_body_axial/10
                
            if i == 0 or i == 10:
                self.phase_bias[i][i+1] = 0
                self.phase_bias[i+1][i] = 0
               
                
            """ Body Antiphase """
            if i < self.n_oscillators_body/2 :
                self.phase_bias[i][i+10] = np.pi
                self.phase_bias[i+10][i] = np.pi

            
    
        for j in range(0,5):
            self.phase_bias[20][j] = 0 #np.pi
        for j in range(10,15):
            self.phase_bias[21][j] = 0 #np.pi
        for j in range(5,10):
            self.phase_bias[22][j] = 0 #np.pi
        for j in range(15,20):
            self.phase_bias[23][j] = 0 #np.pi

        

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        # pylog.warning("Convergence rates must be set")
        
        self.rates[0:20] = 20
        self.rates[20:24] = 20



