# SParameters_for_rectangularwaveguide
Post‑processing pipeline for electromagnetic waveguide simulations. Loads an Elmer/VTK .vtu mesh with Poynting vectors, extracts input/output ports using PyVista, integrates power flow over each port, computes S‑parameters (S11, S21, loss in dB) and visualizes port fields.
# Waveguide S‑Parameter Calculator (PyVista Post‑Processing)

This repository contains a Python script for post‑processing 3D electromagnetic simulations of waveguides (e.g. coplanar waveguides) exported as VTK/VTU files. It uses **PyVista** to extract port surfaces, integrates the **Poynting vector** over these ports, and computes power‑based S‑parameters.

## Features

- Load `.vtu` simulation results containing field data (from Elmer or similar FEM tools)
- Extract the outer surface of the full 3D mesh
- Automatically detect **input** and **output** port planes using geometric thresholds
- Integrate the **Poynting vector** over each port surface using exact cell areas
- Compute:
  - Total input and output power
  - Reflected and transmitted power
  - **S11** and **S21** in linear scale and dB
  - Overall power loss in dB
- Visualize input/output port surfaces colored by the magnitude of the Poynting vector

## Code Overview

The main logic is implemented in the class:

```python
WaveguideSParameterCalculator
