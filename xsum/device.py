"""Default torch device: the GPU when one is available, otherwise the CPU.

Every function that takes a `device` argument defaults to this, so the same code trains, evaluates and samples on a
GPU machine or a CPU-only one without changes."""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
