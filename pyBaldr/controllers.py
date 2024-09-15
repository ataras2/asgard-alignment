
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import signal
import control as ctrl  # Import the control systems library

class PIDController:
    def __init__(self, kp=None, ki=None, kd=None, upper_limit=None, lower_limit=None, setpoint=None, Ts=1.0):
        if kp is None:
            kp = np.zeros(1)
        if ki is None:
            ki = np.zeros(1)
        if kd is None:
            kd = np.zeros(1)
        if lower_limit is None:
            lower_limit = np.zeros(1)
        if upper_limit is None:
            upper_limit = np.ones(1)
        if setpoint is None:
            setpoint = np.zeros(1)

        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.lower_limit = np.array(lower_limit)
        self.upper_limit = np.array(upper_limit)
        self.setpoint = np.array(setpoint)
        self.Ts = Ts  # Sampling time

        size = len(self.kp)
        self.output = np.zeros(size)
        self.integrals = np.zeros(size)
        self.prev_errors = np.zeros(size)

    def process(self, measured):
        measured = np.array(measured)
        size = len(self.setpoint)

        if len(measured) != size:
            raise ValueError(f"Input vector size must match setpoint size: {size}")

        # Check all vectors have the same size
        error_message = []
        for attr_name in ['kp', 'ki', 'kd', 'lower_limit', 'upper_limit']:
            if len(getattr(self, attr_name)) != size:
                error_message.append(attr_name)
        
        if error_message:
            raise ValueError(f"Input vectors of incorrect size: {' '.join(error_message)}")

        if len(self.integrals) != size:
            print("Reinitializing integrals, prev_errors, and output to zero with correct size.")
            self.integrals = np.zeros(size)
            self.prev_errors = np.zeros(size)
            self.output = np.zeros(size)

        for i in range(size):
            error = measured[i] - self.setpoint[i]
            self.integrals[i] += error
            self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

            derivative = error - self.prev_errors[i]
            self.output[i] = (self.kp[i] * error +
                              self.ki[i] * self.integrals[i] +
                              self.kd[i] * derivative)
            self.prev_errors[i] = error

        return self.output

    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)

    # Function to calculate the Z-domain transfer function for a specific mode (index)
    def z_transfer_function(self, mode_index):
        z = sp.symbols('z')
        Ts = self.Ts  # Sampling time

        # Get the PID gains for the selected mode
        Kp = self.kp[mode_index]
        Ki = self.ki[mode_index]
        Kd = self.kd[mode_index]

        # Transfer function in Z-domain: Kp + Ki * Ts / (z - 1) + Kd * (z - 1) / (Ts * z)
        z_transfer_fn = Kp + Ki * Ts / (z - 1) + Kd * (z - 1) / (Ts * z)
        
        return z_transfer_fn

    # Function to convert Z-domain transfer function to S-domain using bilinear transform
    def s_transfer_function(self, mode_index):
        z_transfer_fn = self.z_transfer_function(mode_index)
        z = sp.symbols('z')
        s = sp.symbols('s')
        Ts = self.Ts

        # Bilinear transform: z = (1 + s * Ts / 2) / (1 - s * Ts / 2)
        bilinear_transform = (1 + s * Ts / 2) / (1 - s * Ts / 2)

        # Substitute the bilinear transform into the Z-domain transfer function
        s_transfer_fn = z_transfer_fn.subs(z, bilinear_transform)

        return s_transfer_fn

    # Function to get numerical transfer function coefficients for Bode plot
    def transfer_function_coeffs(self, s_transfer_fn):
        s = sp.symbols('s')
        
        # Get the numerator and denominator of the symbolic transfer function
        num, denom = sp.fraction(sp.simplify(s_transfer_fn))
        
        # Convert the numerator and denominator into polynomials
        num_coeffs = sp.Poly(num, s).all_coeffs()
        denom_coeffs = sp.Poly(denom, s).all_coeffs()
        
        # Convert to float values
        num_coeffs = [float(c) for c in num_coeffs]
        denom_coeffs = [float(c) for c in denom_coeffs]
        
        return num_coeffs, denom_coeffs

    # Function to create a TransferFunction object using control systems library
    def transfer_function_object(self, mode_index):
        # Get the transfer function in S-domain
        s_transfer_fn = self.s_transfer_function(mode_index)
        
        # Get the transfer function coefficients
        num_coeffs, denom_coeffs = self.transfer_function_coeffs(s_transfer_fn)
        
        # Create and return the TransferFunction object using control library
        tf = ctrl.TransferFunction(num_coeffs, denom_coeffs)
        
        return tf

    # Function to plot Bode plot for a specific mode
    def bode_plot(self, mode_index):
        # Get the TransferFunction object
        tf = self.transfer_function_object(mode_index)
        
        # Generate Bode plot
        mag, phase, omega = ctrl.bode(tf, dB=True, Hz=False, deg=True, plot=True)


class LeakyIntegrator:
    def __init__(self, rho=None, lower_limit=None, upper_limit=None, kp=None):
        # If no arguments are passed, initialize with default values
        if rho is None:
            self.rho = []
            self.lower_limit = []
            self.upper_limit = []
            self.kp = []
        else:
            if len(rho) == 0:
                raise ValueError("Rho vector cannot be empty.")
            if len(lower_limit) != len(rho) or len(upper_limit) != len(rho):
                raise ValueError("Lower and upper limit vectors must match rho vector size.")
            if kp is None or len(kp) != len(rho):
                raise ValueError("kp vector must be the same size as rho vector.")

            self.rho = np.array(rho)
            self.output = np.zeros(len(rho))
            self.lower_limit = np.array(lower_limit)
            self.upper_limit = np.array(upper_limit)
            self.kp = np.array(kp)  # kp is a vector now

    def process(self, input_vector):
        input_vector = np.array(input_vector)

        # Error checks
        if len(input_vector) != len(self.rho):
            raise ValueError("Input vector size must match rho vector size.")

        size = len(self.rho)
        error_message = ""

        if len(self.rho) != size:
            error_message += "rho "
        if len(self.lower_limit) != size:
            error_message += "lower_limit "
        if len(self.upper_limit) != size:
            error_message += "upper_limit "
        if len(self.kp) != size:
            error_message += "kp "

        if error_message:
            raise ValueError("Input vectors of incorrect size: " + error_message)

        if len(self.output) != size:
            print(f"output.size() != size.. reinitializing output to zero with correct size")
            self.output = np.zeros(size)

        # Process with the kp vector
        self.output = self.rho * self.output + self.kp * input_vector
        self.output = np.clip(self.output, self.lower_limit, self.upper_limit)

        return self.output

    def reset(self):
        self.output = np.zeros(len(self.rho))

    # Function to calculate the Z-domain transfer function for a specific mode (index)
    def z_transfer_function(self, mode_index):
        z = sp.symbols('z')
        
        # Get the parameters for the selected mode
        rho = self.rho[mode_index]
        kp = self.kp[mode_index]
        
        # Transfer function in Z-domain: H(z) = Kp / (1 - rho * z^-1)
        z_transfer_fn = kp / (1 - rho / z)
        
        return z_transfer_fn

    # Function to convert Z-domain transfer function to S-domain using bilinear transform
    def s_transfer_function(self, mode_index, Ts=1.0):
        z_transfer_fn = self.z_transfer_function(mode_index)
        z = sp.symbols('z')
        s = sp.symbols('s')

        # Bilinear transform: z = (1 + s * Ts / 2) / (1 - s * Ts / 2)
        bilinear_transform = (1 + s * Ts / 2) / (1 - s * Ts / 2)

        # Substitute the bilinear transform into the Z-domain transfer function
        s_transfer_fn = z_transfer_fn.subs(z, bilinear_transform)

        return s_transfer_fn

    # Function to get numerical transfer function coefficients for Bode plot
    def transfer_function_coeffs(self, s_transfer_fn):
        s = sp.symbols('s')
        
        # Get the numerator and denominator of the symbolic transfer function
        num, denom = sp.fraction(sp.simplify(s_transfer_fn))
        
        # Convert the numerator and denominator into polynomials
        num_coeffs = sp.Poly(num, s).all_coeffs()
        denom_coeffs = sp.Poly(denom, s).all_coeffs()
        
        # Convert to float values
        num_coeffs = [float(c) for c in num_coeffs]
        denom_coeffs = [float(c) for c in denom_coeffs]
        
        return num_coeffs, denom_coeffs

    # Function to create a TransferFunction object using control systems library
    def transfer_function_object(self, mode_index, Ts=1.0):
        # Get the transfer function in S-domain
        s_transfer_fn = self.s_transfer_function(mode_index, Ts=Ts)
        
        # Get the transfer function coefficients
        num_coeffs, denom_coeffs = self.transfer_function_coeffs(s_transfer_fn)
        
        # Create and return the TransferFunction object using control library
        tf = ctrl.TransferFunction(num_coeffs, denom_coeffs)
        
        return tf

    # Function to plot Bode plot for a specific mode
    def bode_plot(self, mode_index, Ts=1.0):
        # Get the TransferFunction object
        tf = self.transfer_function_object(mode_index, Ts=Ts)
        
        # Generate Bode plot
        mag, phase, omega = ctrl.bode(tf, dB=True, Hz=False, deg=True, plot=True)


if __name__ == "__main__":

    # Example usage
    pid = PIDController(kp=[1, 2], ki=[0.5, 0.1], kd=[0.01, 0.02], Ts=0.1)

    # Generate TransferFunction object for mode 0
    tf_mode_0 = pid.transfer_function_object(0)
    print(f"Transfer Function for mode 0: \n{tf_mode_0}")

    # Plot Bode plot for mode 0
    pid.bode_plot(0)


    # Example usage
    rho = [0.9, 0.7]
    lower_limit = [-1, -1]
    upper_limit = [1, 1]
    kp = [0.5, 0.3]
    leaky_integrator = LeakyIntegrator(rho=rho, lower_limit=lower_limit, upper_limit=upper_limit, kp=kp)

    # Generate TransferFunction object for mode 0
    tf_mode_0 = leaky_integrator.transfer_function_object(0)
    print(f"Transfer Function for mode 0: \n{tf_mode_0}")

    # Plot Bode plot for mode 0
    leaky_integrator.bode_plot(0)

    # Plot Bode plot for mode 1
    leaky_integrator.bode_plot(1)
