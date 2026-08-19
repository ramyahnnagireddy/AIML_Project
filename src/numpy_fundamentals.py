import numpy as np


def practice():
    # 1D Array
    arr_1d = np.array([1, 2, 3, 4, 5])

    print("1D Array:")
    print(arr_1d)
    print("Shape:", arr_1d.shape)

    # 2D Array
    arr_2d = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("\n2D Array:")
    print(arr_2d)
    print("Shape:", arr_2d.shape)

    # 3D Array
    arr_3d = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    print("\n3D Array:")
    print(arr_3d)
    print("Shape:", arr_3d.shape)


if __name__ == "__main__":
    practice()

    # Broadcasting
    numbers = np.array([10, 20, 30])

    result = numbers + 5

    print("\nBroadcasting:")
    print("Original array:", numbers)
    print("After adding 5:", result)

    # Vectorised Operations
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])

    addition = a + b
    multiplication = a * b
    square = a ** 2

    print("\nVectorised Operations:")
    print("Addition:", addition)
    print("Multiplication:", multiplication)
    print("Square of a:", square)
    # Matrix Multiplication
    matrix_a = np.array([
        [1, 2],
        [3, 4]
    ])

    matrix_b = np.array([
        [5, 6],
        [7, 8]
    ])

    matrix_result = matrix_a @ matrix_b

    print("\nMatrix Multiplication:")
    print("Matrix A:")
    print(matrix_a)

    print("Matrix B:")
    print(matrix_b)

    print("A @ B:")
    print(matrix_result)
    # Statistics from CSV dataset
    data = np.genfromtxt(
        "data/iris.csv",
        delimiter=",",
        skip_header=1,
        usecols=(0, 1, 2, 3)
    )

    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    correlation = np.corrcoef(data, rowvar=False)

    print("\nStatistics from Iris Dataset:")

    print("Mean:")
    print(mean_values)

    print("\nStandard Deviation:")
    print(std_values)

    print("\nCorrelation Matrix:")
    print(correlation)