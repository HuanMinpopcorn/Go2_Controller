import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(t, tf, xf):
    a2 = -3 * xf / tf**2
    a3 = 2 * xf / tf**3
    return - a2 * t**2 - a3 * t**3

# Parameters
tf = 10  # Final time
xf = 5   # Final position

# Generate time points
t = np.linspace(0, tf, 100)
x = cubic_spline(t, tf, xf)

# Plot the cubic spline
plt.figure(figsize=(8, 4))
plt.plot(t, x, label='Cubic Spline')
plt.xlabel('Time (t)')
plt.ylabel('Position (x)')
plt.title('Cubic Spline from [0, 0] to [tf, xf] with Zero Gradient')
plt.grid(True)
plt.legend()
plt.show()
