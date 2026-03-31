# Prepack: Offline Microbatch Packing for Pipeline Parallelism
This repository contains the open-source implementation of offline microbatch packing introduced in _Prepack_, a senior thesis at the University of Michigan. Prepack is a two-stage utility featuring:
- An `OfflinePacker`, which assigns samples into microbatches according to a user-provided packing algorithm and stores the assignments in an index file; and
- A `RuntimeStreamer`, which reads from the index file at training time to stream microbatches into the model, without reexecuting the packing logic.
Prepack is implemented on top of PyTorch. It allows the use of custom datset containers, making it compatible with datasets stored in any format.

## Repository Structure
- `packer.py` contains the core implementation of the `OfflinePacker` and `RuntimeStreamer` classes
- `packing_functions` contains example implementations of packing classes to generate workload-balanced microbatches
- `containers` contains an example implementation of a dataset container for datasets stored in jsonl format

## Installation
Prepack can be installed with the pip package manager using the link to this repository.

## Usage
Users must write a custom class derived from the `OfflinePacker` that implements the `pack_microbatches` method, which is where the packing algorithm should be implemented. The offline component allows specifying the filepath at which to save the packed index file. The streaming component takes the filepath of the index file to read from as input; the same path should be passed to both components.

Once the methods have been customized, the `RuntimeDataset` that is a member variable of the `RuntimeStreamer` method can be iterated over just as the default PyTorch Dataset.
