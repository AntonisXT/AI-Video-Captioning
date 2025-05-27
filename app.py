# =============================================================================
# PYTHON STANDARD LIBRARY IMPORTS - ΠΡΩΤΑ
# =============================================================================
import os
import sys
import asyncio

# =============================================================================
# PYTORCH FIXES - ΠΡΙΝ ΑΠΟ STREAMLIT
# =============================================================================
import torch
torch.classes.__path__ = []

# Environment variables για αποφυγή warnings και conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix για OpenMP conflicts

# Fix για event loop issues
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# =============================================================================
# STREAMLIT IMPORT ΚΑΙ PAGE CONFIG - ΑΜΕΣΩΣ ΜΕΤΑ
# =============================================================================
import streamlit as st

# PAGE CONFIG ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ Η ΠΡΩΤΗ STREAMLIT ΕΝΤΟΛΗ
st.set_page_config(
    page_title="AI Video Captioning",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ΥΠΟΛΟΙΠΑ IMPORTS
# =============================================================================
import json
import time
import tempfile
import shutil
from datetime import datetime
import pandas as pd

# Silent error handling για optional components
try:
    from streamlit_theme import st_theme
    HAS_THEME_COMPONENT = True
except ImportError:
    HAS_THEME_COMPONENT = False

# Import existing modules
from main import VideoCaptioning

# =============================================================================
# ΥΠΟΛΟΙΠΟΣ ΚΩΔΙΚΑΣ ΤΗΣ ΕΦΑΡΜΟΓΗΣ
# =============================================================================

def add_custom_css():
    """Add custom CSS styling"""
    # Ο CSS κώδικάς σου εδώ...
    pass

def initialize_session_state():
    """Initialize session state variables"""
    # Ο κώδικας για session state εδώ...
    pass

# Αφαίρεσε τη συνάρτηση setup_page_config() γιατί το page config είναι ήδη στην αρχή

def main():
    """Main application function"""
    # ΔΕΝ καλείς setup_page_config() εδώ πια!
    add_custom_css()
    initialize_session_state()
    
    # Υπόλοιπος κώδικας της εφαρμογής...

if __name__ == "__main__":
    main()
