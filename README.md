## CS194-26 Proj1: Colorizing the Prokudin-Gorskii Collection

### The Code
All the code required to generate the images can be found in main.py. At the top, you will find one `main()` function, followed by several helper functions.

The `main()` function iterates over all the files in the `data/` directory where the black and white input images are assumed to be stored, and then uses two different scoring methods, SSD and NCC, to align and stack the color channels. It then generates and saves an output image as a .jpg in a directory named `results/`.

The descriptions of all the helper functions are documented within the main.py file itself.

***NOTE***: this will only work if the input images are stored in a `data/` directory relative to the script itself!

### How to run the code
Running the code is very simple:

1. type `python3 main.py` or `python main.py` into the command line
2. there is no second step

Enjoy! :)