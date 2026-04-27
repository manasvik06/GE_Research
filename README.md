# Advancing Convolution Through FFT

This repository contains code and visualizations for a research presentation on speeding up 2D convolution using the Fast Fourier Transform (FFT).

The project compares direct spatial-domain convolution with FFT-based convolution, verifies that both produce the same output, and benchmarks their runtime performance.

## Project Overview

Convolution is widely used in image processing, signal processing, deep learning, and medical imaging. A direct implementation slides a kernel across an image and computes a local dot product at every pixel.

For an `N × N` image and a `K × K` kernel, direct convolution has complexity:

```text
O(N²K²)
