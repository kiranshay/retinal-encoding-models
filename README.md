# Retinal Ganglion Cell Encoding Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-red.svg)](https://retinal-encoding.streamlit.app)

> Interactive demo of how retinal neurons encode visual information into spike trains

## Overview

This app demonstrates the computational models neuroscientists use to predict how **retinal ganglion cells (RGCs)** respond to visual stimuli. It implements two classic encoding models:

- **Linear-Nonlinear (LN) Model** — convolves stimulus with a spatial receptive field and temporal filter, then applies a softplus nonlinearity
- **Generalized Linear Model (GLM)** — extends the LN model with spike history dependence, capturing refractory periods and adaptation

## What You Can Explore

- **Receptive fields** — center-surround structure (ON-center, OFF-center, ON-OFF) modeled as difference-of-Gaussians
- **Temporal dynamics** — biphasic temporal filters showing fast excitation and slower inhibition
- **Stimulus responses** — how different visual stimuli (white noise, moving bars, gratings) drive neural activity
- **Spike generation** — converting continuous firing rates to discrete spike trains via Poisson process
- **Model comparison** — LN vs GLM side-by-side, showing how spike history improves predictions

## Live Demo

**[Try it on Streamlit](https://retinal-encoding.streamlit.app)**

## Running Locally

```bash
git clone https://github.com/kiranshay/retinal-ganglion-cell-encoding-models.git
cd retinal-ganglion-cell-encoding-models
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

The core encoding equation:

```
λ(t) = f(∫ K(τ) · s(t-τ) dτ + ∫ h(τ) · r(t-τ) dτ)
```

Where K is the spatiotemporal receptive field, s is the stimulus, h is the spike history filter, and f is a nonlinearity.

**Note:** This is an educational interactive demo using synthetic stimuli and analytically defined model components. It does not train on or validate against real neural recordings.

## Limitations

- Models use hand-tuned parameters, not fitted to real data
- No real retinal recording data is loaded or compared against
- CNN/deep learning models referenced in theory are not implemented in this demo
- Information-theoretic quantities described in the theory tab are not computed

## References

- Chichilnisky, E.J. (2001). A simple white noise analysis of neuronal light responses. *Network: Computation in Neural Systems*
- Pillow, J.W. et al. (2008). Spatio-temporal correlations and visual signalling in a complete neuronal population. *Nature*
- Field, G.D. & Chichilnisky, E.J. (2007). Information processing in the primate retina. *Annual Review of Neuroscience*

## License

MIT License

---

Built by [Kiran Shay](https://kiranshay.github.io) · Johns Hopkins University
