outputs:

The operation was a success!
Mean time: 3.83374 ms

nvcc -lineinfo  reverse_blocks_shared.cu -o 1.exe

compute-sanitizer --tool memcheck ./1.exe
========= COMPUTE-SANITIZER
The operation was a success!
Mean time: 33.0728 ms
========= ERROR SUMMARY: 0 errors

compute-sanitizer --tool synccheck ./1.exe
========= COMPUTE-SANITIZER
The operation was a success!
Mean time: 49.5191 ms
========= ERROR SUMMARY: 0 errors