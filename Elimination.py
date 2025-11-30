
def main():
    
    # Homework matrices
    augmented_matrix = [[1, 0, 2, 1], [2, -1, 3, -1], [4, 1, 8, 2]]
    augmented_matrix_copy = [[1, 0, 2, 1], [2, -1, 3, -1], [4, 1, 8, 2]]

    matrix = [[1, -1, 0], [-2, 2, -1], [0, 1, -2]]
    matrix_copy = [[1, -1, 0], [-2, 2, -1], [0, 1, -2]]

    # Test matrices
    aug_matrix_textbook_example1 = [[1, -2, 0, -4], [3, 1, -2, 1], [0, -2, 2, 2]]
    aug_matrix_textbook_example2 = [[1, -1, 0, 2], [-2, 2, -1, -1], [0, 1, -2, 6]]
    aug_matrix_textbook_example3 = [[1, 0, 2, 1], [2, -1, 3, -1], [4, 1, 8, 2]]
    
    matrix_textbook_example1 = [[1, -1, 0], [2, 0, 4], [0, 2, -1]]
    matrix_textbook_example2 = [[9, -4, 2], [-3, 0, 6], [3, 1, 3]]
    matrix_textbook_example3 = [[1, 4, 0], [0, 2, 6], [-1, 0, 1]]
    matrix_textbook_example4 = [[3, 4, 5], [6, 9, 7], [12, 8, 10]]
    matrix_textbook_example5 = [[1, 3], [2, 5]]

    print("\nGauss Jordan Elimination")
    print("Solution vector: ", gauss_jordan_elimination(augmented_matrix))

    print("\nMatrix Inversion")
    print("Inverse matrix: ", get_inverse_matrix(matrix))

    print("\nGaussian Elimination")
    print("Solution vector: ", gaussian_elimination(augmented_matrix_copy))

    print("\nMatrix Determinant")
    print("Determinant of matrix: ", get_matrix_determinant(matrix_copy))

def gauss_jordan_elimination(augmented_matrix):

    # Defining the max number of rows and columns for clarity
    num_rows = len(augmented_matrix)
    num_cols = len(augmented_matrix[0])

    # Elimination only on the left-hand side of the augmented matrix
    for col in range(num_cols - 1):
        diagonal_row = col
        highest_mag = abs(augmented_matrix[diagonal_row][col])
        swap_row_index = diagonal_row

        # Swap step
        for row in range(diagonal_row, num_rows):
            if abs(augmented_matrix[row][col]) > abs(highest_mag):
                highest_mag = abs(augmented_matrix[row][col])
                swap_row_index = row
        if swap_row_index != diagonal_row:
                augmented_matrix[diagonal_row], augmented_matrix[swap_row_index] = augmented_matrix[swap_row_index], augmented_matrix[diagonal_row]

        # Divide step
        pivot_value = augmented_matrix[diagonal_row][col]
        for i in range(num_cols):
            augmented_matrix[diagonal_row][i] /= pivot_value

        # Subtract step
        for row in range(num_rows):
            if row != diagonal_row:
                factor = augmented_matrix[row][col]
                for j in range(num_cols):
                    augmented_matrix[row][j] -= factor * augmented_matrix[diagonal_row][j]

    # Extracting the solution variables from the final augmented matrix
    solution_vector = [0] * num_rows
    for row in range(num_rows):
        solution_vector[row] = augmented_matrix[row][-1]

    return solution_vector

def get_inverse_matrix(matrix):

    # Defining the max number of rows and columns for clarity
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Creating the identity matrix
    identity = [[0] * num_rows for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_rows):
            if row == col:
                identity[row][col] = 1
            else:
                identity[row][col] = 0

    # Combining the matrix and its identity
    aug_matr = [matrix[row] + identity[row] for row in range(num_rows)]
    new_cols = len(aug_matr[0])

    # Elimination on the left-hand side of the augmented matrix
    for col in range(num_cols):
        diagonal_row = col
        highest_mag = abs(aug_matr[diagonal_row][col])
        swap_row_index = diagonal_row

        # Swap step
        for row in range(diagonal_row, num_rows):
            if abs(aug_matr[row][col]) > abs(highest_mag):
                highest_mag = abs(aug_matr[row][col])
                swap_row_index = row
        if swap_row_index != diagonal_row:
                aug_matr[diagonal_row], aug_matr[swap_row_index] = aug_matr[swap_row_index], aug_matr[diagonal_row]

        # Divide step
        pivot_value = aug_matr[diagonal_row][col]
        for i in range(new_cols):
            aug_matr[diagonal_row][i] /= pivot_value

        # Subtract step
        for row in range(num_rows):
            if row != diagonal_row:
                factor = aug_matr[row][col]
                for j in range(new_cols):
                    aug_matr[row][j] -= (factor * aug_matr[diagonal_row][j])

    # Extracting the inverse matrix from the final augmented matrix
    inverse_matrix = [[0] * num_rows for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_rows):
            if aug_matr[row][num_cols + col] == 0:      # Prevents -0.0
                inverse_matrix[row][col] = abs(aug_matr[row][num_cols + col])
            else:
                inverse_matrix[row][col] = aug_matr[row][num_cols + col]

    return inverse_matrix

def gaussian_elimination(augmented_matrix):

    # Defining the max number of rows and columns for clarity
    num_rows = len(augmented_matrix)
    num_cols = len(augmented_matrix[0])

    # Elimination only on the left-hand side of the augmented matrix
    for col in range(num_cols - 1):

        diagonal_row = col
        highest_mag = abs(augmented_matrix[diagonal_row][col])
        swap_row_index = diagonal_row

        # Swap step
        for row in range(diagonal_row, num_rows):
            if abs(augmented_matrix[row][col]) > abs(highest_mag):
                highest_mag = abs(augmented_matrix[row][col])
                swap_row_index = row
        if swap_row_index != diagonal_row:
                augmented_matrix[diagonal_row], augmented_matrix[swap_row_index] = augmented_matrix[swap_row_index], augmented_matrix[diagonal_row]

        # Subtract step
        for row in range(diagonal_row, num_rows):
            if row != diagonal_row:
                factor = augmented_matrix[row][col] / augmented_matrix[diagonal_row][col]
                for j in range(num_cols):
                    augmented_matrix[row][j] -= factor * augmented_matrix[diagonal_row][j]
    
    solution_vector = [0] * num_rows

    # Solving for unknowns using back-substitution
    for row in range(num_rows - 1, -1, -1):  # Iterates from the last row to the first
        known_terms_sum = 0
        for col in range(row + 1, num_rows):
            known_terms_sum += augmented_matrix[row][col] * solution_vector[col]
        solution_vector[row] = (augmented_matrix[row][-1] - known_terms_sum) / augmented_matrix[row][row]

    return solution_vector

def get_matrix_determinant(matrix):

    # Defining the max number of rows and columns for clarity
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Row swap counter
    row_swaps = 0

    # Elimination operations on every row and column of matrix
    for col in range(num_cols):

        diagonal_row = col
        highest_mag = abs(matrix[diagonal_row][col])
        swap_row_index = diagonal_row

        # Swap step
        for row in range(diagonal_row, num_rows):
            if abs(matrix[row][col]) > abs(highest_mag):
                highest_mag = abs(matrix[row][col])
                swap_row_index = row
        if swap_row_index != diagonal_row:
                matrix[diagonal_row], matrix[swap_row_index] = matrix[swap_row_index], matrix[diagonal_row]
                row_swaps += 1

        # Subtract step
        for row in range(diagonal_row, num_rows):
            if row != diagonal_row:
                factor = matrix[row][col] / matrix[diagonal_row][col]
                for j in range(num_cols):
                    matrix[row][j] -= factor * matrix[diagonal_row][j]

    # Storing every value on the diagonal
    diagonal_values = [0] * num_rows
    for row in range(num_rows):
        for col in range(num_cols):
            if row == col:
                diagonal_values[row] = matrix[row][col]

    # Getting the product of diagonal values
    diagonal_product = 1
    for value in diagonal_values:
        diagonal_product *= value

    # Calculating the determinant of the matrix
    determinant = ((-1) ** row_swaps) * (diagonal_product)

    return determinant

main()
