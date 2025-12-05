import argparse
import struct
import os

def read_binary_and_write_text(input_binary_filepath: str, output_text_filepath: str):
    """
    Reads integers from a binary file and writes them to a text file, one integer per line.
    Assumes integers are 4-byte unsigned (uint32_t C type) and little-endian.
    """
    try:
        with open(input_binary_filepath, "rb") as binary_file, \
             open(output_text_filepath, "w") as text_file:
            while True:
                # Read 4 bytes at a time
                four_bytes = binary_file.read(4)
                if not four_bytes:
                    break # EOF

                # Unpack as a little-endian unsigned integer
                val = struct.unpack("<I", four_bytes)[0]
                text_file.write(str(val) + "\n")
        print(f"Successfully converted '{input_binary_filepath}' to '{output_text_filepath}'")
    except FileNotFoundError:
        print(f"Error: Input file '{input_binary_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Read a binary file of uint32_t and write integers to a text file.")

    parser.add_argument(
        "input_binary_filepath",
        type=str,
        help="Path to the input binary file (e.g., random-10.bin)"
    )

    args = parser.parse_args()

    # Determine output filename
    # Remove .bin and add .txt, or just append .txt if no extension
    base_name = os.path.basename(args.input_binary_filepath)
    file_name_without_ext, ext = os.path.splitext(base_name)
    if ext.lower() == ".bin":
        output_text_filepath = f"output-{file_name_without_ext}.txt"
    else:
        output_text_filepath = f"output-{base_name}.txt"


    read_binary_and_write_text(args.input_binary_filepath, output_text_filepath)

if __name__ == "__main__":
    main()
