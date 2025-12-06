# CompArch-HW4
HW4 code for computer architecture

## How to generate your data:
Don't add data files to git. The count=10 example is in git for clarity, but others
are too large for git :C
```
cd data
python gen_data.py --count=1000
python gen_data.py --count=10000
python gen_data.py --count=100000
python gen_data.py --count=1000000
python gen_data.py --count=10000000
python gen_data.py --count=100000000  # Will take ~60s to run, generates ~1GB file
```
Every time you run gen_data, a .txt file and 2 .bin files are created with the same random ints in the same order. .bin is used because mmap() requires it (for efficiency purposes, otherwise loading in and writing out data takes a lot of time). 

When running the `sorter` program, it overwrites the .bin file (in place modification), which is why we make 2 .bin files (so you can still have the original input file after running `sorter`)

## How to run the program
2 options (parallel or merge)
```
make
./sorter -f data/random-100000000.bin -type parallel
./sorter -f data/random-100000000.bin -type merge
```

## Where the output is
```
sorted_data/...
```


## How to run the Nvidia GPU Program
Only tested for Windows GTX 1000 series.
This does not modify in place but makes a new copy of the data

Go to windows search bar and look for "x64 Native Tools Command Prompt for VS 2022" (or 2019)
If you don't have anything, download Visual Studio 2022 or 2019 (different IDE than VSCode).
Once you have it, open it and run `nvcc` in this terminal. Then, you can run the executable output anywhere
like normal.
```
nvcc -arch=sm_61 -o sorter gpu_sort_bitonic.cu
gpu_sorter.exe data/random-1000.bin output_bin.bin
```