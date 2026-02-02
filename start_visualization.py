#!/usr/bin/env python3
"""
Example script showing how to use the Visualizer class.
Run this after executing the simulator to generate visualizations from CSV files.
"""

from pathlib import Path
from src.viz import Visualizer

# Create visualizer
viz = Visualizer(output_path=Path("output"))

# Option 1: Generate only static plot (fast)
# viz.draw_all(animate=False)

# Option 2: Generate static plot + animation (slower)
viz.draw_all(animate=True, fps=30, speedup=3.0, linger_seconds=2.0)

print("\n✓ Done! Check the output/ folder for visualizations.")
