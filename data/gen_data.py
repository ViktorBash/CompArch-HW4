"""
Generate data that can be used.

The numbers must be a
uint32_t C type
0 to 2^32-1

Run on your machine:
python gen_data.py --count=1000
python gen_data.py --count=10000
python gen_data.py --count=100000
python gen_data.py --count=1000000
python gen_data.py --count=10000000
python gen_data.py --count=100000000  # Will take ~60s to run, generates ~10GB file
"""

import argparse
import random

BOTTOM_INT_RANGE = 0
TOP_INT_RANGE = 4294967295  # 2^32 - 1


def gen_random_dist(count: int):
    """
    Steps:
    Open a file called "data.txt"
    Write an integer to every single line
    """
    with open(f"random-{count}.txt", "w") as file:
        for _ in range(count):
            val = random.randint(BOTTOM_INT_RANGE, TOP_INT_RANGE)
            file.write(str(val) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Type of data to make")

    parser.add_argument(
        "--distribution",
        type=str,
        choices=["random"],
        default="random",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1_000_000,
    )

    args = parser.parse_args()

    print(f"Data Dist: {args.distribution}, Count: {args.count:,}")

    if args.distribution == "random":
        gen_random_dist(args.count)

if __name__ == "__main__":
    main()