import numpy as np
import argparse
from scipy.sparse import random as sparse_random, save_npz

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate and save a sparse random matrix.")
    parser.add_argument('--n', type=int, default=5000, help='Number of rows in the matrix')
    parser.add_argument('--cols', type=int, default=128, help='Number of columns in the matrix')
    parser.add_argument('--density', type=float, default=0.05, help='Density of non-zero elements')

    # Parse arguments
    args = parser.parse_args()

    # Generate the sparse random matrix
    matrix = sparse_random(args.n, args.cols, density=args.density, format='csr', dtype=float)

    # Save the sparse matrix in compressed .npz format
    save_npz('../../data/SBM/SBM-_feature.npz', matrix)

    print("Sparse matrix generated and saved successfully.")

if __name__ == "__main__":
    main()
