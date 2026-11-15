# Droplet Image Processing

A Python-based image-processing algorithm developed for the geometrical analysis of experimental droplets in electrowetting studies.

## Overview

This project implements an image-processing approach to identify droplet boundaries from experimental images and calculate geometrical quantities, particularly droplet volume.

The algorithm uses grayscale thresholding, Sobel edge detection, baseline detection, and interpolation-based contour tracking.

## Image Processing Methodology

1. **Image acquisition:** Loads an experimental droplet image.
2. **Grayscale conversion:** Converts the input image into grayscale using a custom grayscale threshold to distinguish the droplet from its surroundings.
3. **Top-point detection:** Identifies the uppermost point of the droplet using threshold conditions.
4. **Edge detection:** Uses manually implemented Sobel operators to extract image edges.
5. **Baseline detection:** Identifies the substrate baseline using horizontal edge projection.
6. **Contour tracking:** Tracks the left and right droplet boundaries using interpolation and adaptive searching.
7. **Volume calculation:** Estimates the droplet volume by treating horizontal cross-sections as circular disks.

## Volume Calculation

The droplet volume is estimated by dividing the droplet into thin horizontal circular slices.

The volume is calculated as:

**V ≈ π × Σ(rᵧ² × Δy)**

where:

- V = Droplet volume
- rᵧ = Radius of the droplet at image row y
- Δy = Physical height represented by one pixel

**Note:** The current implementation calculates:

**V = π × Σ(rᵧ²)**

This is a pixel-based volume estimate. Physical volume requires calibration to convert pixel dimensions into physical lengths.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Usage

The main function is:

`get(image_path, top_threshold, gray_threshold)`

- `image_path`: Path to the experimental droplet image.
- `top_threshold`: Gradient threshold for detecting the droplet's top point.
- `gray_threshold`: Grayscale threshold used for image segmentation.

## Project Status

Development version. Image-processing parameters may require adjustment depending on image quality, illumination, and experimental conditions.

## Example Results

### Original Experimental Image

![Input Droplet](sample_images/input_droplet.jpg)

### Detected Edges and Baseline

![Processed Droplet](results/detected_edge and baseline.png)
