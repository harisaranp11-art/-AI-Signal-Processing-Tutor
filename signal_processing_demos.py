# AI Signal Processing Tutor — verified demonstrations
# Run in Google Colab or Jupyter. Dependencies: numpy, matplotlib.

import numpy as np
import matplotlib.pyplot as plt

def demo_sampling():
    f, fs, duration = 5, 50, 1
    t = np.linspace(0, duration, 1000, endpoint=False)
    x = np.sin(2*np.pi*f*t)
    ts = np.arange(0, duration, 1/fs)
    xs = np.sin(2*np.pi*f*ts)
    assert len(ts) == 50
    plt.figure(figsize=(8, 3))
    plt.plot(t, x, label="Original 5 Hz signal")
    plt.stem(ts, xs, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Samples")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("Sampling")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    return {"sample_count": len(ts), "sampling_rate_hz": fs, "signal_frequency_hz": f}

def demo_aliasing():
    f_signal, fs = 9, 12
    t = np.arange(0, 1, 1/fs)
    x = np.sin(2*np.pi*f_signal*t)
    # For a sampled real sinusoid, 9 Hz sampled at 12 Hz aliases to |9-12| = 3 Hz.
    t_dense = np.linspace(0, 1, 1000, endpoint=False)
    plt.figure(figsize=(8, 3))
    plt.plot(t_dense, np.sin(2*np.pi*f_signal*t_dense), alpha=.6, label="Original 9 Hz")
    plt.stem(t, x, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Samples at 12 Hz")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("Aliasing: 9 Hz sampled at 12 Hz")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    return {"original_hz": f_signal, "sampling_hz": fs, "alias_frequency_hz": abs(f_signal-fs)}

def demo_fft():
    fs, duration = 100, 1
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    amplitude = 2*np.abs(X)/len(x)
    amplitude[0] /= 2
    if len(x) % 2 == 0: amplitude[-1] /= 2
    peaks = freqs[np.argsort(amplitude)[-3:]]
    plt.figure(figsize=(8, 3))
    plt.plot(freqs, amplitude)
    plt.xlim(0, 30); plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
    plt.title("FFT magnitude spectrum"); plt.grid(True); plt.tight_layout(); plt.show()
    assert np.isclose(amplitude[np.argmin(abs(freqs-5))], 1.0, atol=1e-10)
    assert np.isclose(amplitude[np.argmin(abs(freqs-15))], 0.5, atol=1e-10)
    return {"peak_5_hz_amplitude": float(amplitude[np.argmin(abs(freqs-5))]),
            "peak_15_hz_amplitude": float(amplitude[np.argmin(abs(freqs-15))])}

def demo_low_pass():
    rng = np.random.default_rng(7)
    fs, duration = 200, 2
    t = np.arange(0, duration, 1/fs)
    clean = np.sin(2*np.pi*5*t)
    noisy = clean + 0.6*rng.normal(size=len(t))
    # Simple moving-average FIR low-pass demonstration.
    taps = 9
    kernel = np.ones(taps)/taps
    filtered = np.convolve(noisy, kernel, mode="same")
    plt.figure(figsize=(8, 3))
    plt.plot(t, noisy, alpha=.45, label="Noisy input")
    plt.plot(t, filtered, label="Moving-average output")
    plt.plot(t, clean, linewidth=2, label="Clean 5 Hz reference")
    plt.xlim(0, 0.8); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.title("Low-pass filtering (moving average)"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.show()
    assert len(filtered) == len(noisy)
    return {"input_noise_std": float(np.std(noisy-clean)),
            "filtered_error_std": float(np.std(filtered-clean)),
            "filter_taps": taps}

def demo_convolution():
    x = np.array([1, 2, 3])
    h = np.array([1, 1])
    y = np.convolve(x, h, mode="full")
    expected = np.array([1, 3, 5, 3])
    assert np.array_equal(y, expected)
    plt.figure(figsize=(8, 3))
    plt.stem(np.arange(len(y)), y, basefmt=" ")
    plt.xlabel("n"); plt.ylabel("y[n]"); plt.title("Discrete convolution")
    plt.grid(True); plt.tight_layout(); plt.show()
    return {"x": x.tolist(), "h": h.tolist(), "convolution": y.tolist()}

def demo_noise():
    rng = np.random.default_rng(42)
    fs, duration = 100, 1
    t = np.arange(0, duration, 1/fs)
    clean = np.sin(2*np.pi*5*t)
    noise = 0.3*rng.normal(size=len(t))
    observed = clean + noise
    measured_noise_std = float(np.std(observed-clean))
    plt.figure(figsize=(8, 3))
    plt.plot(t, clean, label="Clean signal")
    plt.plot(t, observed, alpha=.7, label="Signal + Gaussian noise")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("Noise demonstration")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    assert abs(measured_noise_std-0.3) < 0.1
    return {"target_noise_std": 0.3, "measured_noise_std": measured_noise_std}

if __name__ == "__main__":
    for name, fn in [
        ("Sampling", demo_sampling), ("Aliasing", demo_aliasing),
        ("FFT", demo_fft), ("Low-pass filter", demo_low_pass),
        ("Convolution", demo_convolution), ("Noise", demo_noise)]:
        print(name, fn())
