# CompArch-HW4
HW4 code for computer architecture

## How to generate your data:
Don't add data files to git. The count=10 example is in git for clarity, but others
are too large for git :C
```
python gen_data.py --count=1000
python gen_data.py --count=10000
python gen_data.py --count=100000
python gen_data.py --count=1000000
python gen_data.py --count=10000000
python gen_data.py --count=100000000  # Will take ~60s to run, generates ~10GB file
```

## How to run the program
```
make
./sorting data/random-10.txt
```

## Where the output is
```
sorted_data/random-10.txt
```
