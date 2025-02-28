"""
Utility functions for optimizing memory usage with PyTorch models.
This helps reduce VRAM consumption and allows models to run faster.
"""

import torch
import gc
import os
import psutil

def print_gpu_memory():
    """Print current GPU memory usage for monitoring."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU Device {i}: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    else:
        print("No GPU available.")

def free_gpu_memory():
    """Free up unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Freed GPU memory.")
    else:
        print("No GPU available to free.")

def print_cpu_memory():
    """Print current CPU memory usage for this process."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # Convert bytes to MB
    print(f"CPU Memory usage: {mem:.2f} MB")

def free_cpu_memory():
    """Force garbage collection to free up CPU memory."""
    gc.collect()
    print("Forced garbage collection for CPU memory.")

if __name__ == "__main__":
    print("GPU Memory Info:")
    print_gpu_memory()
    print("\nCPU Memory Info:")
    print_cpu_memory()
    # Optionally free memory
    free_gpu_memory()
    free_cpu_memory()
