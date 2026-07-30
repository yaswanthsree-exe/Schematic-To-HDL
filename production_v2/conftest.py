"""Put production_v2/ on sys.path so tests can import pattern_engine and hdl_gen
the same way app_final.py does (Streamlit adds the script's directory itself)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
