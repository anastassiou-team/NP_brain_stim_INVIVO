"""
File paths configuration for Rembado et al. 2026.
"""
import os

# Base paths — resolved relative to *this* file so scripts work
# regardless of the current working directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input paths
DATA_PATH = os.path.join(BASE_DIR, 'data', 'units')
WF_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'waveform_features')
CONNECTIVITY_PATH = os.path.join(BASE_DIR, 'data', 'connectivity')

# Output paths
FIGURES_OUTPUT = os.path.join(BASE_DIR, 'figures')

# Ensure output directory exists
os.makedirs(FIGURES_OUTPUT, exist_ok=True)
