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
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="Retinal Encoding Models",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional styling system
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background: transparent;
        border-radius: 12px;
        padding: 0 24px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.08);
        color: #667eea;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        display: block;
    }

    /* Subsection Headers */
    .subsection-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #334155;
        margin: 2rem 0 1rem 0;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
    }

    /* Concept Card */
    .concept-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .concept-card h5 {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .concept-card p, .concept-card li {
        color: #1e293b;
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }

    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, #ede9fe 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .highlight-box p {
        color: #4338ca;
        font-weight: 500;
        margin: 0;
    }

    /* Key point callout */
    .key-point {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .key-point-icon {
        font-size: 1.25rem;
        flex-shrink: 0;
    }

    .key-point p {
        color: #065f46;
        font-weight: 500;
        margin: 0;
        line-height: 1.6;
    }

    /* Definition list styling */
    .def-item {
        display: flex;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f1f5f9;
    }

    .def-term {
        font-weight: 600;
        color: #667eea;
        min-width: 140px;
        flex-shrink: 0;
    }

    .def-desc {
        color: #1e293b;
        line-height: 1.5;
    }

    /* Algorithm steps */
    .algo-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px dashed #e2e8f0;
    }

    .algo-step:last-child {
        border-bottom: none;
    }

    .step-num {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .step-content {
        color: #1e293b;
        line-height: 1.6;
    }

    /* Parameter cards */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .param-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
    }

    .param-card h6 {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #667eea;
        margin: 0 0 0.5rem 0;
    }

    .param-card p {
        color: #334155;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Cell type cards */
    .cell-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: all 0.2s ease;
    }

    .cell-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .cell-card h4 {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
    }

    .cell-card .cell-type {
        font-style: italic;
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .cell-card .props {
        color: #1e293b;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Metric container */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
    }

    .metric-container h3 {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .metric-container p {
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 500;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
    }

    /* Expander styling */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] span {
        color: #1e293b !important;
    }

    [data-testid="stExpander"] strong {
        color: #0f172a !important;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }

    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">👁️ Retinal Ganglion Cell Encoding Models</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Decoding how neurons transform visual scenes into neural code</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## 🎛️ Model Parameters")

cell_type = st.sidebar.selectbox(
    "Cell Type",
    ["ON-center", "OFF-center", "ON-OFF"],
    help="Different retinal ganglion cell types have different receptive field structures"
)

rf_size = st.sidebar.slider("Receptive Field Size", 5, 25, 15, step=2)
surround_ratio = st.sidebar.slider("Center-Surround Ratio", 1.5, 4.0, 2.5, step=0.1)
temporal_decay = st.sidebar.slider("Temporal Decay (ms)", 10, 100, 50)
noise_level = st.sidebar.slider("Neural Noise Level", 0.0, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧬 Spike Generation")
base_rate = st.sidebar.slider("Base Firing Rate (Hz)", 5, 50, 20)
refractory_ms = st.sidebar.slider("Refractory Period (ms)", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 GLM Parameters")
history_length = st.sidebar.slider("Spike History Length (ms)", 10, 100, 50)
history_weight = st.sidebar.slider("History Weight", -1.0, 0.0, -0.3, step=0.1)


# ==================== Helper Functions ====================

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
    tau1 = decay_ms
    tau2 = decay_ms * 2
    h = (t/tau1) * np.exp(-t/tau1) - 0.5 * (t/tau2) * np.exp(-t/tau2)
    return h / np.max(np.abs(h))


def create_spike_history_filter(length_ms, weight, dt=1):
    """Create a spike history filter (post-spike suppression)."""
    t = np.arange(0, length_ms, dt)
    # Exponential decay with initial refractory period
    h = weight * np.exp(-t / (length_ms / 3))
    h[:2] = -10  # Strong refractory period
    return h


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
            if (t // 20) % 2 == 0:
                stimulus[t, center-spot_size:center+spot_size,
                        center-spot_size:center+spot_size] = 1
        return stimulus
    else:  # Drifting Grating
        stimulus = np.zeros((n_frames, size, size))
        x = np.linspace(0, 4*np.pi, size)
        for t in range(n_frames):
            phase = t * 0.1
            grating = np.sin(x + phase)
            stimulus[t] = np.tile(grating, (size, 1))
        return stimulus


def predict_response_ln(stimulus, rf, temporal_filter, noise_level):
    """Predict neural response using LN model."""
    n_frames = stimulus.shape[0]

    # Spatial filtering
    spatial_response = np.array([
        np.sum(stimulus[t] * rf) for t in range(n_frames)
    ])

    # Temporal filtering
    response = np.convolve(spatial_response, temporal_filter, mode='same')

    # Nonlinearity (softplus)
    response = np.log1p(np.exp(response))

    # Add noise
    response += np.random.randn(len(response)) * noise_level * np.std(response)

    return np.maximum(response, 0)


def predict_response_glm(stimulus, rf, temporal_filter, history_filter, noise_level, base_rate):
    """Predict neural response using GLM with spike history."""
    n_frames = stimulus.shape[0]
    dt = 5  # ms per frame

    # Spatial filtering
    spatial_response = np.array([
        np.sum(stimulus[t] * rf) for t in range(n_frames)
    ])

    # Temporal filtering
    stim_component = np.convolve(spatial_response, temporal_filter, mode='same')

    # Initialize response and spike history
    response = np.zeros(n_frames)
    spike_train = np.zeros(n_frames)
    history_len = len(history_filter)

    for t in range(n_frames):
        # Stimulus component
        rate = base_rate + stim_component[t] * 20

        # Add spike history effect
        if t > 0:
            start_idx = max(0, t - history_len)
            hist_contribution = np.sum(spike_train[start_idx:t] * history_filter[-(t-start_idx):])
            rate += hist_contribution * 10

        # Add noise
        rate += np.random.randn() * noise_level * 5

        # Ensure non-negative
        rate = max(0, rate)
        response[t] = rate

        # Generate spikes (Poisson process)
        prob = 1 - np.exp(-rate * dt / 1000)
        if np.random.rand() < prob:
            spike_train[t] = 1

    return response, spike_train


def generate_poisson_spikes(rate_trace, dt_ms=5, refractory_ms=2):
    """Generate Poisson spike train from firing rate trace."""
    n_frames = len(rate_trace)
    spike_train = np.zeros(n_frames)
    last_spike = -refractory_ms - 1

    for t in range(n_frames):
        time_since_spike = (t * dt_ms) - last_spike
        if time_since_spike > refractory_ms:
            rate = max(0, rate_trace[t])
            prob = 1 - np.exp(-rate * dt_ms / 1000)
            if np.random.rand() < prob:
                spike_train[t] = 1
                last_spike = t * dt_ms

    return spike_train


# ==================== Main Content ====================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Receptive Field",
    "⏱️ Temporal Dynamics",
    "🎬 Stimulus Response",
    "⚡ Spike Raster",
    "🔬 Model Comparison",
    "📚 Theory"
])

# ==================== Tab 1: Receptive Field ====================
with tab1:
    st.markdown('<p class="section-header">Spatial Receptive Field</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
<div class="concept-card">
    <h5>👁️ What is a Receptive Field?</h5>
    <p>A receptive field describes which region of visual space a neuron responds to.
    Retinal ganglion cells have a characteristic <strong>center-surround</strong> structure:</p>
    <ul>
        <li><strong>ON-center cells</strong>: Excited by light in center, inhibited by surround</li>
        <li><strong>OFF-center cells</strong>: Inhibited by light in center, excited by surround</li>
        <li><strong>ON-OFF cells</strong>: Respond to both light onset and offset</li>
    </ul>
    <p>This structure helps detect <em>edges and contrast</em> in visual scenes.</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="highlight-box">
<p>💡 The center-surround structure is created by the <strong>difference of Gaussians (DoG)</strong> model, mimicking the actual neural circuitry of retinal horizontal and bipolar cells.</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        rf = create_receptive_field(rf_size, cell_type, surround_ratio)

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = LinearSegmentedColormap.from_list('custom', ['#3b82f6', '#f8fafc', '#ef4444'])
        im = ax.imshow(rf, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f'{cell_type} Receptive Field', fontsize=14, fontweight='bold', color='#1e293b')
        ax.set_xlabel('Horizontal position (pixels)', fontsize=11, color='#475569')
        ax.set_ylabel('Vertical position (pixels)', fontsize=11, color='#475569')
        cbar = plt.colorbar(im, ax=ax, label='Response weight')
        cbar.ax.tick_params(labelsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Cell type comparison
    st.markdown('<div class="subsection-header">📊 Cell Type Comparison</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="cell-card">
    <h4>☀️ ON-Center</h4>
    <div class="cell-type">Responds to light increments</div>
    <div class="props">
        <strong>Function:</strong><br>
        • Detects bright objects on dark backgrounds<br>
        • Active during daytime vision<br>
        • ~50% of ganglion cells
    </div>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="cell-card">
    <h4>🌙 OFF-Center</h4>
    <div class="cell-type">Responds to light decrements</div>
    <div class="props">
        <strong>Function:</strong><br>
        • Detects dark objects on light backgrounds<br>
        • Active during shadow detection<br>
        • ~50% of ganglion cells
    </div>
</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="cell-card">
    <h4>⚡ ON-OFF</h4>
    <div class="cell-type">Responds to any change</div>
    <div class="props">
        <strong>Function:</strong><br>
        • Motion detection<br>
        • Edge detection<br>
        • Transient responses
    </div>
</div>
        """, unsafe_allow_html=True)


# ==================== Tab 2: Temporal Dynamics ====================
with tab2:
    st.markdown('<p class="section-header">Temporal Filter</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
<div class="concept-card">
    <h5>⏱️ Temporal Dynamics</h5>
    <p>Neurons don't respond instantaneously - they integrate visual information over time.
    The temporal filter describes this integration:</p>
    <ul>
        <li><strong>Fast excitatory phase</strong>: Initial response to stimulus onset</li>
        <li><strong>Slower inhibitory phase</strong>: Adaptation that reduces sustained response</li>
    </ul>
    <p>This biphasic structure helps neurons respond to <em>changes</em> rather than static images.</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="key-point">
    <div class="key-point-icon">🎯</div>
    <p><strong>Biological basis:</strong> The biphasic filter emerges from the interaction of fast glutamate receptors and slower inhibitory feedback circuits in the retina.</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        temporal = create_temporal_filter(200, temporal_decay)
        t = np.arange(len(temporal))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, temporal, color='#667eea', linewidth=2.5)
        ax.fill_between(t, temporal, 0, where=temporal > 0, alpha=0.3, color='#667eea', label='Excitation')
        ax.fill_between(t, temporal, 0, where=temporal < 0, alpha=0.3, color='#ef4444', label='Inhibition')
        ax.axhline(y=0, color='#94a3b8', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)', fontsize=12, color='#475569')
        ax.set_ylabel('Filter weight', fontsize=12, color='#475569')
        ax.set_title('Temporal Filter', fontsize=14, fontweight='bold', color='#1e293b')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Spike history filter
    st.markdown('<div class="subsection-header">🔄 Spike History Filter (GLM)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
<div class="concept-card">
    <h5>📉 Post-Spike Dynamics</h5>
    <p>After a neuron fires, its excitability changes:</p>
    <ul>
        <li><strong>Absolute refractory period</strong>: Cannot fire (~1-2 ms)</li>
        <li><strong>Relative refractory period</strong>: Reduced excitability (~10-50 ms)</li>
        <li><strong>Adaptation</strong>: Gradual recovery to baseline</li>
    </ul>
    <p>The GLM captures these effects with a spike history filter.</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        history = create_spike_history_filter(history_length, history_weight)
        t_hist = np.arange(len(history))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_hist, history, color='#ef4444', linewidth=2.5)
        ax.fill_between(t_hist, history, 0, alpha=0.3, color='#ef4444')
        ax.axhline(y=0, color='#94a3b8', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time since spike (ms)', fontsize=12, color='#475569')
        ax.set_ylabel('Effect on firing rate', fontsize=12, color='#475569')
        ax.set_title('Spike History Filter', fontsize=14, fontweight='bold', color='#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================== Tab 3: Stimulus Response ====================
with tab3:
    st.markdown('<p class="section-header">Stimulus-Response Prediction</p>', unsafe_allow_html=True)

    stim_type = st.selectbox("Select Stimulus Type",
                              ["White Noise", "Moving Bar", "Flashing Spot", "Drifting Grating"])

    # Generate data
    n_frames = 200
    stimulus = generate_stimulus(stim_type, rf_size, n_frames)
    rf = create_receptive_field(rf_size, cell_type, surround_ratio)
    temporal = create_temporal_filter(50, temporal_decay)
    response = predict_response_ln(stimulus, rf, temporal, noise_level)

    col1, col2 = st.columns([1, 1])

    # ---- Stimulus Animation Playback ----
    st.markdown("### 🎬 Stimulus Animation")
    st.markdown("""
<div class="highlight-box">
<p>🎥 <strong>Interactive Playback:</strong> Use the slider to scrub through stimulus frames and see how the neural response corresponds to each moment in time. The response trace highlights the current time point.</p>
</div>
    """, unsafe_allow_html=True)

    # Frame selection slider
    current_frame = st.slider("Select Frame", 0, n_frames - 1, 0, key="stim_frame_slider")

    col_anim1, col_anim2 = st.columns([1, 2])

    with col_anim1:
        # Display current stimulus frame
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(stimulus[current_frame], cmap='gray', vmin=-2, vmax=2)
        ax.set_title(f'Frame {current_frame} ({current_frame * 5} ms)', fontsize=14, fontweight='bold', color='#1e293b')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_anim2:
        # Display firing rate with current position highlighted
        fig, ax = plt.subplots(figsize=(10, 4))
        time = np.arange(n_frames) * 5
        ax.plot(time, response, color='#667eea', linewidth=1.5, alpha=0.7)
        ax.fill_between(time, response, alpha=0.2, color='#667eea')

        # Highlight current time point
        current_time = current_frame * 5
        ax.axvline(x=current_time, color='#e74c3c', linewidth=2, linestyle='--', label='Current frame')
        ax.scatter([current_time], [response[current_frame]], color='#e74c3c', s=100, zorder=5, edgecolors='white', linewidths=2)

        ax.set_xlabel('Time (ms)', fontsize=12, color='#475569')
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12, color='#475569')
        ax.set_title('Neural Response (current position marked)', fontsize=14, fontweight='bold', color='#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Current frame stats
    st.markdown(f"""
<div class="metric-container" style="text-align: center; margin-top: 1rem;">
    <p>At frame {current_frame} ({current_frame * 5} ms): <strong>Firing Rate = {response[current_frame]:.1f} Hz</strong></p>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Static Overview ----
    with col1:
        st.markdown("### 📺 Stimulus Frame Samples")

        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        frames = [0, n_frames//4, n_frames//2, 3*n_frames//4]
        for i, f in enumerate(frames):
            axes[i].imshow(stimulus[f], cmap='gray', vmin=-2, vmax=2)
            axes[i].set_title(f'Frame {f}', fontsize=10, fontweight='bold')
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📈 Full Firing Rate Trace")

        fig, ax = plt.subplots(figsize=(10, 3))
        time = np.arange(n_frames) * 5
        ax.plot(time, response, color='#667eea', linewidth=1.5)
        ax.fill_between(time, response, alpha=0.3, color='#667eea')
        ax.set_xlabel('Time (ms)', fontsize=12, color='#475569')
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12, color='#475569')
        ax.set_title('LN Model Prediction', fontsize=14, fontweight='bold', color='#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
<div class="metric-container">
    <h3>{np.mean(response):.1f} Hz</h3>
    <p>Mean Firing Rate</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div class="metric-container">
    <h3>{np.max(response):.1f} Hz</h3>
    <p>Peak Response</p>
</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
<div class="metric-container">
    <h3>{np.std(response):.1f}</h3>
    <p>Response Variability</p>
</div>
        """, unsafe_allow_html=True)

    with col4:
        sparseness = np.sum(response > np.mean(response)) / len(response)
        st.markdown(f"""
<div class="metric-container">
    <h3>{sparseness:.2f}</h3>
    <p>Sparseness Index</p>
</div>
        """, unsafe_allow_html=True)


# ==================== Tab 4: Spike Raster ====================
with tab4:
    st.markdown('<p class="section-header">Spike Raster Plot</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>⚡ <strong>From rates to spikes:</strong> Real neurons communicate through discrete action potentials. We convert the continuous firing rate into spike trains using a Poisson process with a refractory period.</p>
</div>
    """, unsafe_allow_html=True)

    n_trials = st.slider("Number of trials", 5, 30, 15)

    # Generate multiple trials
    stimulus = generate_stimulus("White Noise", rf_size, n_frames)
    rf = create_receptive_field(rf_size, cell_type, surround_ratio)
    temporal = create_temporal_filter(50, temporal_decay)

    all_spike_trains = []
    all_responses = []

    for _ in range(n_trials):
        response = predict_response_ln(stimulus, rf, temporal, noise_level)
        spikes = generate_poisson_spikes(response * base_rate / 10, dt_ms=5, refractory_ms=refractory_ms)
        all_spike_trains.append(spikes)
        all_responses.append(response)

    all_spike_trains = np.array(all_spike_trains)
    all_responses = np.array(all_responses)

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

        # Raster plot
        time = np.arange(n_frames) * 5
        for trial in range(n_trials):
            spike_times = time[all_spike_trains[trial] > 0]
            axes[0].scatter(spike_times, np.ones_like(spike_times) * trial + 1,
                           marker='|', s=30, color='#1e293b', linewidths=1)

        axes[0].set_ylabel('Trial', fontsize=12, color='#475569')
        axes[0].set_title('Spike Raster', fontsize=14, fontweight='bold', color='#1e293b')
        axes[0].set_ylim(0.5, n_trials + 0.5)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # PSTH (Peri-Stimulus Time Histogram)
        psth = np.sum(all_spike_trains, axis=0) / n_trials * (1000 / 5)  # Convert to Hz
        psth_smooth = gaussian_filter(psth, sigma=2)

        axes[1].bar(time, psth, width=5, color='#94a3b8', alpha=0.6, label='Raw PSTH')
        axes[1].plot(time, psth_smooth, color='#667eea', linewidth=2, label='Smoothed')
        axes[1].set_xlabel('Time (ms)', fontsize=12, color='#475569')
        axes[1].set_ylabel('Firing Rate (Hz)', fontsize=12, color='#475569')
        axes[1].set_title('PSTH (Peri-Stimulus Time Histogram)', fontsize=14, fontweight='bold', color='#1e293b')
        axes[1].legend(frameon=True, loc='upper right')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # ISI histogram
        all_isis = []
        for trial in range(n_trials):
            spike_times = np.where(all_spike_trains[trial] > 0)[0] * 5
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                all_isis.extend(isis)

        if len(all_isis) > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(all_isis, bins=30, color='#667eea', alpha=0.7, edgecolor='white')
            ax.axvline(x=refractory_ms, color='#ef4444', linestyle='--', linewidth=2, label=f'Refractory ({refractory_ms}ms)')
            ax.set_xlabel('Inter-Spike Interval (ms)', fontsize=11, color='#475569')
            ax.set_ylabel('Count', fontsize=11, color='#475569')
            ax.set_title('ISI Distribution', fontsize=13, fontweight='bold', color='#1e293b')
            ax.legend(frameon=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Statistics
            total_spikes = np.sum(all_spike_trains)
            mean_rate = total_spikes / (n_trials * n_frames * 5 / 1000)
            cv_isi = np.std(all_isis) / np.mean(all_isis) if len(all_isis) > 1 else 0

            st.markdown(f"""
<div class="concept-card">
    <h5>📊 Spike Statistics</h5>
    <p><strong>Total spikes:</strong> {int(total_spikes)}</p>
    <p><strong>Mean rate:</strong> {mean_rate:.1f} Hz</p>
    <p><strong>CV(ISI):</strong> {cv_isi:.2f}</p>
    <p><em>CV ≈ 1 indicates Poisson-like variability</em></p>
</div>
            """, unsafe_allow_html=True)


# ==================== Tab 5: Model Comparison ====================
with tab5:
    st.markdown('<p class="section-header">Model Comparison: LN vs GLM</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="highlight-box">
<p>🔬 <strong>Why compare models?</strong> The LN model is simple but ignores spike history. The GLM adds post-spike dynamics, capturing adaptation and refractory effects that shape real neural responses.</p>
</div>
    """, unsafe_allow_html=True)

    # Generate data for both models
    stimulus = generate_stimulus("White Noise", rf_size, 200)
    rf = create_receptive_field(rf_size, cell_type, surround_ratio)
    temporal = create_temporal_filter(50, temporal_decay)
    history = create_spike_history_filter(history_length, history_weight)

    # LN model
    response_ln = predict_response_ln(stimulus, rf, temporal, noise_level)
    spikes_ln = generate_poisson_spikes(response_ln * base_rate / 10, dt_ms=5, refractory_ms=refractory_ms)

    # GLM
    response_glm, spikes_glm = predict_response_glm(stimulus, rf, temporal, history, noise_level, base_rate)

    time = np.arange(200) * 5

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 LN Model")
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

        axes[0].plot(time, response_ln, color='#667eea', linewidth=1.5)
        axes[0].fill_between(time, response_ln, alpha=0.3, color='#667eea')
        axes[0].set_ylabel('Rate', fontsize=11)
        axes[0].set_title('Firing Rate', fontsize=12, fontweight='bold')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        spike_times_ln = time[spikes_ln > 0]
        axes[1].scatter(spike_times_ln, np.ones_like(spike_times_ln), marker='|', s=100, color='#1e293b')
        axes[1].set_xlabel('Time (ms)', fontsize=11)
        axes[1].set_ylabel('Spikes', fontsize=11)
        axes[1].set_ylim(0.5, 1.5)
        axes[1].set_yticks([])
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
<div class="param-card">
    <h6>LN Model Stats</h6>
    <p>Spikes: {int(np.sum(spikes_ln))} | Mean rate: {np.mean(response_ln):.1f}</p>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 GLM (with History)")
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

        axes[0].plot(time, response_glm, color='#764ba2', linewidth=1.5)
        axes[0].fill_between(time, response_glm, alpha=0.3, color='#764ba2')
        axes[0].set_ylabel('Rate', fontsize=11)
        axes[0].set_title('Firing Rate', fontsize=12, fontweight='bold')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        spike_times_glm = time[spikes_glm > 0]
        axes[1].scatter(spike_times_glm, np.ones_like(spike_times_glm), marker='|', s=100, color='#1e293b')
        axes[1].set_xlabel('Time (ms)', fontsize=11)
        axes[1].set_ylabel('Spikes', fontsize=11)
        axes[1].set_ylim(0.5, 1.5)
        axes[1].set_yticks([])
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
<div class="param-card">
    <h6>GLM Stats</h6>
    <p>Spikes: {int(np.sum(spikes_glm))} | Mean rate: {np.mean(response_glm):.1f}</p>
</div>
        """, unsafe_allow_html=True)

    # Model comparison explanation
    st.markdown('<div class="subsection-header">🔍 Key Differences</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="cell-card">
    <h4>🎯 Predictive Power</h4>
    <div class="props">
        • LN: ~60-70% variance explained<br>
        • GLM: ~80-90% variance explained<br>
        • History effects add ~10-20% accuracy
    </div>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="cell-card">
    <h4>⚙️ Complexity</h4>
    <div class="props">
        • LN: 2 components (RF + nonlinearity)<br>
        • GLM: 3+ components (+ spike history)<br>
        • More parameters = risk of overfitting
    </div>
</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="cell-card">
    <h4>🧬 Biological Realism</h4>
    <div class="props">
        • LN: Misses adaptation effects<br>
        • GLM: Captures refractory periods<br>
        • GLM: Models burst suppression
    </div>
</div>
        """, unsafe_allow_html=True)


# ==================== Tab 6: Theory ====================
with tab6:
    st.markdown('<p class="section-header">📚 Theoretical Background</p>', unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🧠 The Neural Encoding Problem</div>', unsafe_allow_html=True)

    with st.expander("**What is Neural Encoding?**", expanded=True):
        st.markdown("""
Neural encoding refers to how sensory information is transformed into patterns of neural activity.
In the retina, this involves transforming a continuous stream of photons into discrete spike trains.
        """)

        st.markdown("""
<div class="highlight-box">
<p>🎯 <strong>The fundamental question:</strong> Given a stimulus s(t), what is the probability of observing a spike train r(t)?</p>
</div>
        """, unsafe_allow_html=True)

        st.latex(r"P(\text{spike train} | \text{stimulus}) = ?")

    with st.expander("**Linear-Nonlinear (LN) Model**", expanded=True):
        st.markdown("The LN model is a classic approach that decomposes neural encoding into two stages:")

        st.markdown("""
<div class="algo-step">
    <div class="step-num">1</div>
    <div class="step-content"><strong>Linear filtering:</strong> Convolve stimulus with receptive field and temporal filter</div>
</div>
<div class="algo-step">
    <div class="step-num">2</div>
    <div class="step-content"><strong>Static nonlinearity:</strong> Apply a nonlinear function (e.g., softplus, exponential)</div>
</div>
        """, unsafe_allow_html=True)

        st.latex(r"\lambda(t) = f\left(\int K(\tau) \cdot s(t-\tau) d\tau\right)")

        st.markdown("""
<div class="def-item"><span class="def-term">K(τ)</span><span class="def-desc">Spatiotemporal receptive field (linear filter)</span></div>
<div class="def-item"><span class="def-term">s(t)</span><span class="def-desc">Visual stimulus</span></div>
<div class="def-item"><span class="def-term">f(·)</span><span class="def-desc">Static nonlinearity (converts to firing rate)</span></div>
<div class="def-item"><span class="def-term">λ(t)</span><span class="def-desc">Instantaneous firing rate</span></div>
        """, unsafe_allow_html=True)

    with st.expander("**Generalized Linear Model (GLM)**", expanded=True):
        st.markdown("The GLM extends the LN model by adding spike history dependence:")

        st.latex(r"\lambda(t) = f\left(\int K(\tau) \cdot s(t-\tau) d\tau + \int h(\tau) \cdot r(t-\tau) d\tau\right)")

        st.markdown("""
<div class="concept-card">
    <h5>📊 Key Addition: Spike History Filter h(τ)</h5>
    <p>The history filter captures:</p>
    <ul>
        <li><strong>Refractory period:</strong> Strong suppression immediately after a spike</li>
        <li><strong>Adaptation:</strong> Gradual decrease in excitability over 10-100ms</li>
        <li><strong>Burst dynamics:</strong> Some cells show facilitation after spikes</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="key-point">
    <div class="key-point-icon">✓</div>
    <p><strong>Why GLM?</strong> By accounting for spike history, the GLM captures temporal correlations in spike trains that the LN model misses, improving prediction accuracy by 10-20%.</p>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**Information Theory in Neural Coding**", expanded=True):
        st.markdown("We can quantify how much visual information neurons transmit using **Shannon's information theory**:")

        st.latex(r"I(S; R) = H(R) - H(R|S)")

        st.markdown("""
<div class="param-grid">
    <div class="param-card">
        <h6>H(R) - Response Entropy</h6>
        <p>Total variability in neural response. More variable = higher entropy.</p>
    </div>
    <div class="param-card">
        <h6>H(R|S) - Noise Entropy</h6>
        <p>Variability due to intrinsic noise. Measured from repeated trials.</p>
    </div>
    <div class="param-card">
        <h6>I(S;R) - Mutual Information</h6>
        <p>Information transmitted about stimulus. Measured in bits/second.</p>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="highlight-box">
<p>📊 <strong>Typical values:</strong> Retinal ganglion cells transmit 2-4 bits/spike, with information rates of 10-100 bits/second depending on the stimulus.</p>
</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">📖 Key References</div>', unsafe_allow_html=True)

    st.markdown("""
| Paper | Key Contribution |
|-------|------------------|
| **Chichilnisky (2001)** | White noise analysis and receptive field estimation |
| **Pillow et al. (2008)** | GLM framework for neural populations |
| **Field & Chichilnisky (2007)** | Information theory in retinal coding |
| **Keat et al. (2001)** | Spike timing precision in RGCs |
    """)


# ==================== Footer ====================
st.markdown("""
<div class="footer">
    <p><strong>👁️ Retinal Ganglion Cell Encoding Models</strong></p>
    <p>
        <a href="https://github.com/kiranshay/retinal-ganglion-cell-encoding-models" target="_blank">GitHub</a> ·
        <a href="https://kiranshay.github.io" target="_blank">Portfolio</a> ·
        <a href="mailto:kiranshay123@gmail.com">Contact</a>
    </p>
    <p style="font-size: 0.85rem; color: #94a3b8;">Computational Neuroscience • Neural Encoding • Johns Hopkins University</p>
</div>
""", unsafe_allow_html=True)

# Session state for interactivity
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
