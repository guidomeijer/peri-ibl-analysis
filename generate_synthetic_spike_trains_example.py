import numpy as np
import matplotlib.pyplot as plt
import elephant
import quantities as pq
import neo

# Define the time axis
duration = 10.0 * pq.s  # Duration of the spike train
time_resolution = 1 * pq.ms  # Time resolution
t = np.arange(0, duration.magnitude, time_resolution.magnitude / 1000) * pq.s

# Define a slowly fluctuating firing rate (e.g., sinusoidal modulation)
rate = 20 + 10 * np.sin(2 * np.pi * t / duration)
rate = rate * pq.Hz  # Convert the rate to Hz

# Create a neo.AnalogSignal object for the rate
rate_signal = neo.AnalogSignal(rate, 
                               sampling_rate=1/time_resolution, 
                               units=pq.Hz)

# Generate an inhomogeneous Poisson spike train
spike_train = elephant.spike_train_generation.inhomogeneous_poisson_process(rate_signal)

# Convert the spike train to a numpy array for plotting
spike_times = spike_train.magnitude

# Plot the rate and the spike train
plt.figure(figsize=(10, 6))

# Plot the firing rate
plt.subplot(2, 1, 1)
plt.plot(t, rate, label='Firing rate')
plt.xlabel('Time (s)')
plt.ylabel('Rate (Hz)')
plt.legend()

# Plot the spike train
plt.subplot(2, 1, 2)
plt.eventplot(spike_times, orientation='horizontal')
plt.xlabel('Time (s)')
plt.ylabel('Spike Train')

plt.tight_layout()
plt.show()
