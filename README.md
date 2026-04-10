# CMU 11-868: Large Language Model Systems

This repository contains my solutions and notes for CMU **11-868: Large Language Model Systems**.

## Links

- Course: [llmsystem2025spring](https://llmsystem.github.io/llmsystem2025spring/)
- Homework guide: [llmsystemhomework](https://llmsystem.github.io/llmsystemhomework/)
- Notes: [blog.mindorigin.top/AI/llmsys](https://blog.mindorigin.top/AI/llmsys)

## Repository Layout

- `llmsys_hw1` - autodiff and MiniTorch basics
- `llmsys_hw2` - CUDA kernels and tensor ops
- `llmsys_hw3` - language model training
- `llmsys_hw4` - CUDA kernel optimization
- `llmsys_hw5` - data parallel and pipeline parallel
- `llmsys_hw6` - DeepSpeed and SGLang experiments

## Notes

These are assignment-level pitfalls I ran into. They are worth checking first before assuming your own implementation is wrong.

- `llmsys_hw1`: gradient buffers must be initialized to zero. If not, later backward tests can fail in non-obvious ways.
- `llmsys_hw2`: the CUDA matmul launch can be wrong if `m` and `p` are used in the wrong grid order. The correct launch is:

```cuda
dim3 gridDims(
    (p + threadsPerBlock - 1) / threadsPerBlock,
    (m + threadsPerBlock - 1) / threadsPerBlock,
    batch
);
```

- `llmsys_hw4`: `import pycuda.autoinit` can interfere with PyTorch CUDA setup. If that happens, remove it and initialize CUDA from PyTorch instead:

```python
import torch

if torch.cuda.is_available():
    _ = torch.tensor([1.0]).cuda()
```
