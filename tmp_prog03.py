import sys
import math
import threading
import numpy as np
import time
from functools import wraps

from typing import Tuple

from numpy.linalg import LinAlgError
from numpy.random import default_rng

sys.setrecursionlimit(10**7)
threading.stack_size(2**27)

global FLOATING_POINT_OPERATIONS
FLOATING_POINT_OPERATIONS = 0


def timeit(func):
    is_evaluating = False

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        nonlocal is_evaluating
        if is_evaluating:
            return func(*args, **kwargs)
        else:
            start_time = time.perf_counter()
            is_evaluating = True
            try:
                result = func(*args, **kwargs)
            finally:
                is_evaluating = False

            end_time = time.perf_counter()
            total_time = end_time - start_time
            # print(f'{func.__name__} took {total_time:.4f} seconds')
            return result

    return timeit_wrapper


def traditional_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    global FLOATING_POINT_OPERATIONS

    final_matrix = []

    for row_idx_A in range(A.shape[0]):
        final_row = []

        for col_idx_B in range(B.shape[1]):
            row_col_product = 0
            col_elem_B = 0

            for row_elem_A in range(A.shape[0]):
                row_col_product += \
                    A[row_idx_A][row_elem_A]*B[col_elem_B][col_idx_B]

                FLOATING_POINT_OPERATIONS += 1

                col_elem_B = col_elem_B + 1

            FLOATING_POINT_OPERATIONS += 1
            final_row.append(row_col_product)
        final_matrix.append(final_row)

    return np.array(final_matrix)


def strassen_2D(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    P1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
    P2 = (A[1][0] + A[1][1]) * B[0][0]
    P3 = A[0][0] * (B[0][1] - B[1][1])
    P4 = A[1][1] * (B[1][0] - B[0][0])
    P5 = (A[0][0] + A[0][1]) * B[1][1]
    P6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
    P7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])

    return np.array([
                [P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]
           ])


def split_matrix_into_quadrants(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(A)
    split_point = size // 2

    top_left = [
        [A[row_idx][col_idx] for col_idx in range(split_point)]
        for row_idx in range(split_point)
    ]

    top_right = [
        [A[row_idx][col_idx] for col_idx in range(split_point, size)]
        for row_idx in range(split_point)
    ]

    bottom_left = [
        [A[row_idx][col_idx] for col_idx in range(split_point)]
        for row_idx in range(split_point, size)
    ]

    bottom_right = [
        [A[row_idx][col_idx] for col_idx in range(split_point, size)]
        for row_idx in range(split_point, size)
    ]

    return np.array(top_left), np.array(top_right), \
        np.array(bottom_left), np.array(bottom_right)


def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    final_matrix = [
        [
            A[row_idx][col_idx] + B[row_idx][col_idx]
            for col_idx in range(len(A[row_idx]))
        ]
        for row_idx in range(len(A))
    ]

    return np.array(final_matrix)


def substract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    final_matrix = [
        [
            A[row_idx][col_idx] - B[row_idx][col_idx]
            for col_idx in range(len(A[row_idx]))
        ]
        for row_idx in range(len(A))
    ]

    return np.array(final_matrix)


def strassen_recursive_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    global FLOATING_POINT_OPERATIONS
    FLOATING_POINT_OPERATIONS += 25

    # Base Case
    if A.shape == (2, 2):
        return strassen_2D(A, B)

    A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(A)
    B_11, B_12, B_21, B_22 = split_matrix_into_quadrants(B)

    P1 = strassen_recursive_algorithm(
        add_matrices(A_11, A_22),
        add_matrices(B_11, B_22)
    )

    P2 = strassen_recursive_algorithm(
        add_matrices(A_21, A_22), B_11
    )

    P3 = strassen_recursive_algorithm(
        A_11, substract_matrices(B_12, B_22)
    )

    P4 = strassen_recursive_algorithm(
        A_22, substract_matrices(B_21, B_11)
    )

    P5 = strassen_recursive_algorithm(
        add_matrices(A_11, A_12), B_22
    )

    P6 = strassen_recursive_algorithm(
        substract_matrices(A_21, A_11),
        add_matrices(B_11, B_12)
    )

    P7 = strassen_recursive_algorithm(
        substract_matrices(A_12, A_22),
        add_matrices(B_21, B_22)
    )

    # Top Left Quadrant
    C_11 = add_matrices(
        P1, add_matrices(substract_matrices(P4, P5), P7))

    # Top Right Quadrant
    C_12 = add_matrices(P3, P5)

    # Bottom Left Quadrant
    C_21 = add_matrices(P2, P4)

    # Bottom Right Quadrant
    C_22 = add_matrices(
        substract_matrices(P1, P2),
        add_matrices(P3, P6)
    )

    final_matrix = []

    for idx in range(len(C_12)):
        # Construct the top of the final matrix
        # Top = Top Left Quadrant + Top Right Quadrant
        final_matrix.append(list(C_11[idx]) + list(C_12[idx]))

    for idx in range(len(C_22)):
        # Construct the bottom of the final matrix
        # Bottom = Bottom Left Quadrant + Bottom Right Quadrant
        final_matrix.append(list(C_21[idx]) + list(C_22[idx]))

    return np.array(final_matrix)


def log2(x: int) -> float:
    return math.log10(x) / math.log10(2)


def is_power_of_two(n: int) -> bool:
    return math.ceil(log2(n)) == math.floor(log2(n))


def check_matrix_is_square(A: np.ndarray) -> bool:
    return A.shape[0] == A.shape[1]


def check_matrix_order_is_power_of_two(A: np.ndarray) -> bool:
    return is_power_of_two(A.shape[0]) and is_power_of_two(A.shape[1])


def check_matrix_mul_condition(A: np.ndarray, B: np.ndarray) -> bool:
    return A.shape[1] == B.shape[0]


def check_matrices_preconditions(A: np.ndarray, B: np.ndarray) -> bool:
    # Ensure matrices are square ones
    assert check_matrix_is_square(A)
    assert check_matrix_is_square(B)

    # Ensure matrices order is a power of two
    assert check_matrix_order_is_power_of_two(A)
    assert check_matrix_order_is_power_of_two(B)

    # Ensure matrix multiplication condition is satisfied
    assert check_matrix_mul_condition(A, B)


def multiply_matrices(A: np.ndarray, B: np.ndarray, l: int) -> np.array:
    # Handle trivial case
    if len(A) == 1 and len(B) == 1:
        return np.array([list(A)[0] * list(B)[0]])

    check_matrices_preconditions(A, B)

    matrix_order = A.shape[0]
    k = log2(matrix_order)

    if k <= l:
        # print("Using Traditional Algorithm.")
        return traditional_algorithm(A, B)
    elif k > l:
        # print("Using Recurent Strassen Algorithm.")
        return strassen_recursive_algorithm(A, B)

def get_identity_matrix(shape: int) -> np.array:
    return np.identity(shape)


def inverse_2D(matrix: np.array, l: int) -> np.array:
    global FLOATING_POINT_OPERATIONS
    FLOATING_POINT_OPERATIONS += 14

    A_11 = matrix[0][0]
    A_12 = matrix[0][1]
    A_21 = matrix[1][0]
    A_22 = matrix[1][1]

    A_11_inv = inverse(A_11, l)
    S_22 = A_22 - A_21 * A_11_inv * A_12
    S_22_inv = inverse(S_22, l)
    B_11 = A_11_inv * (1 + A_12 * S_22_inv * A_21 * A_11_inv)
    B_12 = -A_11_inv * A_12 * S_22_inv
    B_21 = -S_22_inv * A_21 * A_11_inv
    B_22 = S_22_inv

    return np.array([[B_11, B_12], [B_21, B_22]])


def inverse(matrix, l):
    global FLOATING_POINT_OPERATIONS
    try:
        x, y = matrix.shape
    except ValueError:
        FLOATING_POINT_OPERATIONS += 1
        if matrix == 0:
            raise LinAlgError("Can't handle zero-like element!")
        else:
            return 1/matrix
    else:
        if matrix.shape == (2, 2):
            return inverse_2D(matrix, l)
        else:
            FLOATING_POINT_OPERATIONS += 4
            A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(matrix)
            i_matrix = get_identity_matrix(A_11.shape[0])

            A_11_inv = inverse(A_11, l)
            S_22 = substract_matrices(
                A_22,
                multiply_matrices(multiply_matrices(A_21, A_11_inv, l), A_12, l)
            )
            S_22_inv = inverse(S_22, l)
            B_11 = multiply_matrices(A_11_inv, (i_matrix + multiply_matrices(multiply_matrices(multiply_matrices(A_12, S_22_inv, l), A_21, l), A_11_inv, l)), l)
            B_12 = multiply_matrices(multiply_matrices(-A_11_inv, A_12, l), S_22_inv, l)
            B_21 = multiply_matrices(multiply_matrices(-S_22_inv, A_21, l), A_11_inv, l)
            B_22 = S_22_inv

            return merge_matrices(B_11, B_12, B_21, B_22)


def merge_matrices(first_matrix, second_matrix, third_matrix, forth_matrix):
    first_row = np.concatenate((first_matrix, second_matrix), axis=1)
    second_row = np.concatenate((third_matrix, forth_matrix), axis=1)
    return np.concatenate((first_row, second_row), axis=0)


def get_zeros_matrix(shape: int) -> np.ndarray:
    return np.zeros((shape, shape))


def LU_factorization_2D(matrix: np.ndarray, l:int) -> Tuple[np.ndarray, np.ndarray]:
    A_11 = matrix[0][0]
    A_12 = matrix[0][1]
    A_21 = matrix[1][0]
    A_22 = matrix[1][1]

    L_11, U_11 = LU_factorization(A_11, l)

    U_11_inv = inverse(U_11, l)
    L_21 = A_21 * U_11_inv
    L_11_inv = inverse(L_11, l)
    U_12 = L_11_inv * A_12
    S = A_22 - A_21 * U_11_inv * L_11_inv * A_12
    Ls, Us = LU_factorization(S, l)
    U_22, L_22 = Us, Ls

    return (
        np.array([[L_11, 0], [L_21, L_22]]),
        np.array([[U_11, U_12], [0, U_22]])
    )


@timeit
def LU_factorization(matrix: np.ndarray, l: int):
    global FLOATING_POINT_OPERATIONS

    if isinstance(matrix, np.int32) or isinstance(matrix, np.float64):
        return (np.array(1), np.array(matrix))
    elif matrix.shape == (2, 2):
        return LU_factorization_2D(matrix, l)
    else:
        A_11, A_12, A_21, A_22 = split_matrix_into_quadrants(matrix)

        L_11, U_11 = LU_factorization(A_11, l)
        U_11_inv = inverse(U_11, l)
        L_21 = multiply_matrices(A_21, U_11_inv, l)
        L_11_inv = inverse(L_11, l)
        U_12 = multiply_matrices(L_11_inv, A_12, l)
        S = substract_matrices(
            A_22,
            multiply_matrices(multiply_matrices(multiply_matrices(A_21, U_11_inv, l), L_11_inv, l), A_12, l)
        )
        Ls, Us = LU_factorization(S, l)
        U_22, L_22 = Us, Ls

        return (
            merge_matrices(L_11, get_zeros_matrix(L_11.shape[0]), L_21, L_22),
            merge_matrices(U_11, U_12, get_zeros_matrix(U_11.shape[0]), U_22)
        )


def calculate_determinant_based_on_LU_matrices(L_result: np.ndarray, U_result: np.ndarray):
    return np.prod(L_result.diagonal()) * np.prod(U_result.diagonal())
