# 100 Days of GPU

A personal journey to learn GPU programming by writing CUDA code consistently for 100 days.

This project is inspired by the [100 Days of GPU Challenge](https://github.com/hkproj/100-days-of-gpu).

## Goals

- Build a solid understanding of GPU programming.
- Learn CUDA through small, practical exercises.
- Progress from basic kernels to parallel algorithms.
- Document what I learn each day.
- Prioritize consistency and understanding over perfection.

## Challenge Rules

1. Write GPU-related code every day.
2. Keep each day's work in its own directory.
3. Document the exercise and the concepts learned.
4. Test the code before marking the day as completed.
5. Commit source code, not compiled binaries.
6. Every submitted solution must be understood by me.

## AI Usage Policy

AI may be used as a learning tool, not as a tool that completes the challenge for me.

I may use AI to:

- Explain concepts and error messages.
- Provide hints and learning resources.
- Review code that I have written.
- Help me understand why something does not work.

I will not use AI to:

- Generate complete solutions before I attempt the exercise.
- Submit code that I do not understand.
- Replace the process of researching, debugging, and learning.

Every solution in this repository is written, tested, and understood by me.

## Progress

| Day | Exercise                                                | Concepts                                                                             | Status |
| --: | ------------------------------------------------------- | ------------------------------------------------------------------------------------ | :----: |
| 001 | A single GPU thread writes `42`, and the host prints it | Kernel launch, device memory, synchronization, D2H copy                              |   ✅    |
| 002 | Element-wise vector addition `C = A + B` with `N = 256` | Global thread indexing, grid configuration, bounds checking, result validation       |   ✅    |
| 003 | Vector addition with arbitrary sizes                    | Dynamic grid sizing, ceiling division, excess thread protection, multiple test cases |   ✅    |
| 004 | CUDA error handling with `CUDA_CHECK`                   | Error codes, macros, file and line diagnostics, launch and synchronization errors    |   ✅    |
| 005 | SAXPY implementation and CPU comparison                 | Floating-point operations, CPU reference, numerical tolerance, result comparison     |   ✅    |
| 006 | ReLU, Leaky ReLU, sigmoid, tanh, and hard sigmoid       | Element-wise activations, branching, math functions, CPU–GPU validation              |   ✅    |
| 007 | Vector addition using a grid-stride loop                | Grid-wide stride, thread reuse, multiple elements per thread, scalable data traversal |   ✅    |

## Repository Structure

```text
.
├── day-1/
│   └── helloworld.cu
├── day-2/
│   └── gpu_vector_addition.cu
├── day-3/
│   └── gpu_vector_addition_N.cu
├── day-4/
│   └── cuda_check_error.cu
├── day-5/
│   └── saxypy.cu
├── day-6/
│   └── activation_functions.cu
├── day-7/
│   └── gride-stride_loop.cu
└── README.md
```

## Environment

- CUDA C/C++
- NVIDIA CUDA Toolkit
- Ubuntu on WSL
- NVIDIA GPU

## Motivation

The goal is not to write perfect code from the beginning. The goal is to learn something every day, understand my mistakes, and become progressively better at GPU programming.
