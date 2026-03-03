"""
Retinal Ganglion Cell Encoding Models - Interactive Demo
=========================================================
Streamlit app demonstrating how neural encoding models predict
retinal neuron responses to visual stimuli.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

st.set_page_config(
    page_title="Retinal Encoding Models",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1366F0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7ff 0%, #e8f4ff 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1366F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Retinal Ganglion Cell Encoding Models</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive visualization of how neurons encode visual information</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Model Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["ON-center", "OFF-center", "ON-OFF"],
    help="Different retinal ganglion cell types have different receptive field structures"
)

rf_size = st.sidebar.slider("Receptive Field Size", 5, 25, 15, step=2)
surround_ratio = st.sidebar.slider("Center-Surround Ratio", 1.5, 4.0, 2.5, step=0.1)
temporal_decay = st.sidebar.slider("Temporal Decay (ms)", 10, 100, 50)
noise_level = st.sidebar.slider("Neural Noise Level", 0.0, 0.5, 0.1)

# Helper functions
def create_receptive_field(size, cell_type, surround_ratio):
    """Create a difference-of-Gaussians receptive field."""
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(x, y)

    center_sigma = size / 6
    surround_sigma = center_sigma * surround_ratio

    center = np.exp(-(X**2 + Y**2) / (2 * center_sigma**2))
    surround = np.exp(-(X**2 + Y**2) / (2 * surround_sigma**2))

    if cell_type == "ON-center":
        rf = center - 0.5 * surround
    elif cell_type == "OFF-center":
        rf = -center + 0.5 * surround
    else:  # ON-OFF
        rf = center - 0.3 * surround

    return rf / np.max(np.abs(rf))

def create_temporal_filter(length_ms, decay_ms, dt=1):
    """Create a biphasic temporal filter."""
    t = np.arange(0, length_ms, dt)
    # Biphasic filter: fast excitation followed by slower inhibition
    tau1 = decay_ms
    tau2 = decay_ms * 2
    h = (t/tau1) * np.exp(-t/tau1) - 0.5 * (t/tau2) * np.exp(-t/tau2)
    return h / np.max(np.abs(h))

def generate_stimulus(stim_type, size, n_frames):
    """Generate different types of visual stimuli."""
    if stim_type == "White Noise":
        return np.random.randn(n_frames, size, size)
    elif stim_type == "Moving Bar":
        stimulus = np.zeros((n_frames, size, size))
        bar_width = 3
        for t in range(n_frames):
            pos = int((t / n_frames) * (size + bar_width)) - bar_width
            stimulus[t, :, max(0, pos):min(size, pos + bar_width)] = 1
        return stimulus
    elif stim_type == "Flashing Spot":
        stimulus = np.zeros((n_frames, size, size))
        center = size // 2
        spot_size = size // 4
        for t in range(n_frames):
            if (t // 20) % 2 == 0:  # Flash on/off every 20 frames
                stimulus[t, center-spot_size:center+spot_size,
                        center-spot_size:center+spot_size] = 1
        return stimulus
    else:  # Grating
        stimulus = np.zeros((n_frames, size, size))
        x = np.linspace(0, 4*np.pi, size)
        for t in range(n_frames):
            phase = t * 0.1
            grating = np.sin(x + phase)
            stimulus[t] = np.tile(grating, (size, 1))
        return stimulus

def predict_response(stimulus, rf, temporal_filter, noise_level):
    """Predict neural response using LN model."""
    n_frames = stimulus.shape[0]

    # Spatial filtering
    spatial_response = np.array([
        np.sum(stimulus[t] * rf) for t in range(n_frames)
    ])

    # Temporal filtering
    response = np.convolve(spatial_response, temporal_filter, mode='same')

    # Nonlinearity (rectification + softplus)
    response = np.log1p(np.exp(response))

    # Add noise
    response += np.random.randn(len(response)) * noise_level * np.std(response)

    return np.maximum(response, 0)

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Receptive Field",
    "⏱️ Temporal Dynamics",
    "🎬 Stimulus Response",
    "📊 Information Theory"
])

with tab1:
    st.header("Spatial Receptive Field")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **What is a Receptive Field?**

        A receptive field describes which region of visual space a neuron responds to.
        Retinal ganglion cells have a characteristic **center-surround** structure:

        - **ON-center cells**: Excited by light in center, inhibited by light in surround
        - **OFF-center cells**: Inhibited by light in center, excited by light in surround
        - **ON-OFF cells**: Respond to both light onset and offset

        This structure helps detect edges and contrast in visual scenes.
        """)

    with col2:
        rf = create_receptive_field(rf_size, cell_type, surround_ratio)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(rf, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{cell_type} Receptive Field', fontsize=14, fontweight='bold')
        ax.set_xlabel('Horizontal position (pixels)')
        ax.set_ylabel('Vertical position (pixels)')
        plt.colorbar(im, ax=ax, label='Response weight')
        st.pyplot(fig)
        plt.close()

with tab2:
    st.header("Temporal Filter")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **Temporal Dynamics**

        Neurons don't respond instantaneously - they integrate visual information over time.
        The temporal filter describes this integration:

        - **Fast excitatory phase**: Initial response to stimulus onset
        - **Slower inhibitory phase**: Adaptation that reduces sustained response

        This biphasic structure helps neurons respond to **changes** rather than static images.
        """)

    with col2:
        temporal = create_temporal_filter(200, temporal_decay)
        t = np.arange(len(temporal))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, temporal, 'b-', linewidth=2)
        ax.fill_between(t, temporal, 0, where=temporal > 0, alpha=0.3, color='blue', label='Excitation')
        ax.fill_between(t, temporal, 0, where=temporal < 0, alpha=0.3, color='red', label='Inhibition')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Filter weight', fontsize=12)
        ax.set_title('Temporal Filter', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

with tab3:
    st.header("Stimulus-Response Prediction")

    stim_type = st.selectbox("Select Stimulus Type",
                              ["White Noise", "Moving Bar", "Flashing Spot", "Drifting Grating"])

    # Generate data
    n_frames = 200
    stimulus = generate_stimulus(stim_type, rf_size, n_frames)
    rf = create_receptive_field(rf_size, cell_type, surround_ratio)
    temporal = create_temporal_filter(50, temporal_decay)
    response = predict_response(stimulus, rf, temporal, noise_level)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Stimulus")

        # Show a few frames
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        frames = [0, n_frames//4, n_frames//2, 3*n_frames//4]
        for i, f in enumerate(frames):
            axes[i].imshow(stimulus[f], cmap='gray', vmin=-2, vmax=2)
            axes[i].set_title(f'Frame {f}')
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Predicted Neural Response")

        fig, ax = plt.subplots(figsize=(10, 3))
        time = np.arange(n_frames) * 5  # 5ms per frame
        ax.plot(time, response, 'b-', linewidth=1.5)
        ax.fill_between(time, response, alpha=0.3)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Firing Rate (spikes/s)', fontsize=12)
        ax.set_title('Model Prediction', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Firing Rate", f"{np.mean(response):.1f} Hz")
    with col2:
        st.metric("Max Response", f"{np.max(response):.1f} Hz")
    with col3:
        st.metric("Response Variability", f"{np.std(response):.1f}")
    with col4:
        st.metric("Sparseness", f"{np.sum(response > np.mean(response)) / len(response):.2f}")

with tab4:
    st.header("Information Transmission")

    st.markdown("""
    **How much visual information do retinal neurons transmit?**

    We can quantify information using **Shannon's information theory**:
    - **Entropy (H)**: Total variability in neural response
    - **Noise Entropy (Hn)**: Variability due to noise
    - **Mutual Information (I)**: I = H - Hn (bits per second)
    """)

    # Simulate multiple trials to estimate information
    n_trials = 50
    responses = []

    for _ in range(n_trials):
        stimulus = generate_stimulus("White Noise", rf_size, 100)
        resp = predict_response(stimulus, rf, temporal, noise_level)
        responses.append(resp)

    responses = np.array(responses)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Response histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(responses.flatten(), bins=30, density=True, alpha=0.7, color='steelblue')
        ax.set_xlabel('Firing Rate (Hz)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Response Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        # Trial-to-trial variability
        mean_resp = np.mean(responses, axis=0)
        std_resp = np.std(responses, axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        time = np.arange(len(mean_resp)) * 5
        ax.plot(time, mean_resp, 'b-', linewidth=2, label='Mean')
        ax.fill_between(time, mean_resp - std_resp, mean_resp + std_resp,
                       alpha=0.3, label='± 1 SD')
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title('Trial-to-Trial Variability', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Information metrics
    total_variance = np.var(responses.flatten())
    noise_variance = np.mean(np.var(responses, axis=0))
    signal_variance = total_variance - noise_variance

    # Approximate mutual information (assuming Gaussian)
    if noise_variance > 0:
        snr = signal_variance / noise_variance
        info_rate = 0.5 * np.log2(1 + snr) * 1000 / 5  # bits/second
    else:
        info_rate = 0

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Signal-to-Noise Ratio", f"{snr:.2f}" if noise_variance > 0 else "∞")
    with col2:
        st.metric("Information Rate", f"{info_rate:.1f} bits/s")
    with col3:
        efficiency = min(100, info_rate / 100 * 100)  # Normalized to typical RGC
        st.metric("Coding Efficiency", f"{efficiency:.0f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Retinal Encoding Models</strong> | Built by Kiran Shay</p>
    <p>Johns Hopkins University | Neuroscience & Computer Science</p>
    <p><a href="https://github.com/kiranshay/retinal-encoding-models">GitHub</a> |
       <a href="https://kiranshay.github.io">Portfolio</a></p>
</div>
""", unsafe_allow_html=True)
