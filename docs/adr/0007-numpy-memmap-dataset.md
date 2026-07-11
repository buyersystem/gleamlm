# ADR-0007: numpy memmap Dataset

All tokenized training data is stored as `.npy` files on disk, loaded via `np.load(..., mmap_mode='r')`. The `LMDataset` class provides sliding-window iteration without loading the full dataset into memory.

Before memmap, DataLoader workers each loaded the full dataset, causing 60-70 GB memory spikes on multi-worker setups and freezes on Windows. After memmap, memory footprint is ~1 MB regardless of dataset size.

Consequence: Single-GPU training on 12 GB VRAM is viable for any dataset size. The memmap design is described as the "core data design" of GleamLM.
