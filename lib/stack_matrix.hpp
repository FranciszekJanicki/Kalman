#ifndef STACK_MATRIX_HPP
#define STACK_MATRIX_HPP

#include "common.hpp"
#include <array>
#include <cstddef>
#include <exception>
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

namespace Linalg::Stack {

    template <Arithmetic Value, Size ROWS, Size COLS>
    struct Matrix {
        using Data = std::array<std::array<Value, COLS>, ROWS>;
        using Init = std::initializer_list<std::initializer_list<Value>>;
        using Row = std::array<Value, COLS>;
        using Column = std::array<Value, ROWS>;

        [[nodiscard]] static constexpr Data make_data(Init const init) noexcept
        {
            Data result;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    result[i][j] = *((init.begin() + i)->begin() + j);
                }
            }
            return result;
        }

        constexpr Matrix() noexcept = default;

        constexpr Matrix(Init const init) noexcept : data{make_data(init)}
        {}

        constexpr Matrix(Matrix&& other) noexcept = default;
        constexpr Matrix(Matrix const& other) noexcept = default;

        constexpr Matrix& operator=(Matrix&& other) noexcept = default;
        constexpr Matrix& operator=(Matrix const& other) noexcept = default;

        constexpr ~Matrix() noexcept = default;

        [[nodiscard]] constexpr Row const& operator[](this Matrix const& self, Size const row)
        {
            if (row > ROWS) {
                throw Error{"Out of bounds\n"};
            }
            return self.data[row];
        }

        [[nodiscard]] constexpr Row& operator[](this Matrix& self, Size const row)
        {
            if (row > ROWS) {
                throw Error{"Out of bounds\n"};
            }
            return self.data[row];
        }

        [[nodiscard]] constexpr Value const& operator[](this Matrix const& self, Size const row, Size const col)
        {
            if (row > ROWS || col > COLS) {
                throw Error{"Out of bounds\n"};
            }
            return self.data[row][col];
        }

        [[nodiscard]] constexpr Value& operator[](this Matrix& self, Size const row, Size const col)
        {
            if (row > ROWS || col > COLS) {
                throw Error{"Out of bounds\n"};
            }
            return self.data[row][col];
        }

        [[nodiscard]] constexpr bool operator<=>(this Matrix const& self, Matrix const& other) noexcept = default;

        constexpr void operator=(this Matrix& self, Init const init)
        {
            self.data = make_data(init);
        }

        constexpr void print(this Matrix const& self) noexcept
        {
            fmt::print("[");
            if (!self.data.empty()) {
                for (auto const& row : self.data) {
                    fmt::print("[");
                    if (!row.empty()) {
                        for (auto const col : row) {
                            fmt::print("{}", col);
                            if (col != row.back()) {
                                fmt::print(", ");
                            }
                        }
                    }
                    fmt::print("]");
                    if (row != self.data.back()) {
                        fmt::print(",\n");
                    }
                }
            }
            fmt::print("]\n");
        }

        [[nodiscard]] constexpr bool is_square(this Matrix const& self) noexcept
        {
            return ROWS == COLS;
        }

        [[nodiscard]] constexpr Size rows(this Matrix const& self) noexcept
        {
            return ROWS;
        }

        [[nodiscard]] constexpr Size cols(this Matrix const& self) noexcept
        {
            return COLS;
        }

        [[nodiscard]] Matrix operator+=(this Matrix& self, Matrix const& other) noexcept
        {
            self = matrix_sum(self, other);
            return self;
        }

        [[nodiscard]] Matrix operator-=(this Matrix& self, Matrix const& other) noexcept
        {
            self = matrix_difference(self, other);
            return self;
        }

        [[nodiscard]] Matrix operator*=(this Matrix& self, Matrix const& other)
        {
            try {
                self = matrix_product(self, other);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        [[nodiscard]] Matrix operator*=(this Matrix& self, Value const scale)
        {
            try {
                self = matrix_scale(self, scale);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        [[nodiscard]] Matrix operator/=(this Matrix& self, Value const scale)
        {
            try {
                self = matrix_scale(self, 1 / scale);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        [[nodiscard]] Matrix operator/=(this Matrix& self, Matrix const& other)
        {
            try {
                self = matrix_product(self, matrix_inverse(other));
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        [[nodiscard]] Matrix operator^=(this Matrix& self, Value const power)
        {
            self = matrix_power(self, power);
            return self;
        }

        Data data{};
    };

    template <Arithmetic Value, Size COLS>
    using Row = Matrix<Value, 1, COLS>;

    template <Arithmetic Value, Size ROWS>
    using Column = Matrix<Value, ROWS, 1>;

    template <Arithmetic Value, Size DIMS>
    using Square = Matrix<Value, DIMS, DIMS>;

    template <Arithmetic Value>
    using Scalar = Matrix<Value, 1, 1>;

    template <Arithmetic Value, Size LEFT_ROWS, Size LEFT_COLS_RIGHT_ROWS, Size RIGHT_COLS>
    using Product = Matrix<Value, LEFT_ROWS, RIGHT_COLS>;

    template <Arithmetic Value, Size ROWS, Size COLS>
    using Transpose = Matrix<Value, COLS, ROWS>;

    template <Arithmetic Value, Size DIMS>
    using Minor = Square<Value, DIMS - 1>;

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] constexpr Square<Value, DIMS> make_eye() noexcept
    {
        Square<Value, DIMS> result;
        for (Size i{}; i < DIMS; ++i) {
            for (Size j{}; j < DIMS; ++j) {
                result[i, j] = (i == j) ? Value{1} : Value{0};
            }
        }
        return result;
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS - 1>
    matrix_minor(Square<Value, DIMS> const& matrix, Size const row, Size const column)
    {
        if (row >= DIMS || column >= DIMS) {
            throw Error{"Wrong dimensions\n"};
        }
        if constexpr (DIMS == 1) {
            return matrix;
        }

        if constexpr (DIMS > 1) {
            Size cof_i{};
            Size cof_j{};
            Square<Value, DIMS - 1> result{};
            for (Size i{}; i < DIMS - 1; ++i) {
                for (Size j{}; j < DIMS - 1; ++j) {
                    if (i != row && j != column) {
                        result[cof_i, cof_j++] = matrix[i, j];
                        if (cof_j == DIMS - 1) {
                            cof_j = 0;
                            ++cof_i;
                        }
                    }
                }
            }
            return result;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_complement(Square<Value, DIMS> const& matrix)
    {
        if constexpr (DIMS == 1) {
            return matrix;
        }

        if constexpr (DIMS > 1) {
            Square<Value, DIMS> result{};
            for (Size i{}; i < DIMS; ++i) {
                for (Size j{}; j < DIMS; ++j) {
                    try {
                        result[i, j] = ((i + j) % 2 == 0 ? 1 : -1) * matrix_det(matrix_minor(matrix, i, j));
                    } catch (Error const& error) {
                        throw error;
                    }
                }
            }
            return result;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_adjoint(Square<Value, DIMS> const& matrix)
    {
        try {
            return matrix_transpose(matrix_complement(matrix));
        } catch (Error const& error) {
            error;
        }
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, COLS, ROWS> matrix_transpose(Matrix<Value, ROWS, COLS> const& matrix) noexcept
    {
        Matrix<Value, COLS, ROWS> result{};
        for (Size i{}; i < ROWS; ++i) {
            for (Size j{}; j < COLS; ++j) {
                result[i, j] = matrix[j, i];
            }
        }
        return result;
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Value matrix_det(Square<Value, DIMS> const& matrix)
    {
        if constexpr (DIMS == 1) {
            return matrix[0, 0];
        }

        if constexpr (DIMS == 2) {
            return (matrix[0, 0] * matrix[1, 1]) - (matrix[1, 0] * matrix[0, 1]);
        }

        if constexpr (DIMS > 2) {
            try {
                Value det{};
                for (Size i{}; i < DIMS; ++i) {
                    det += (i % 2 == 0 ? 1 : -1) * matrix[0, i] * matrix_det(matrix_minor(matrix, 0, i));
                }
                return det;
            } catch (Error const& error) {
                throw error;
            }
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_inverse(Square<Value, DIMS> const& matrix)
    {
        try {
            return matrix_scale(matrix_adjoint(matrix), 1 / matrix_det(matrix));
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_upper_triangular(Square<Value, DIMS> const& matrix)
    {
        try {
            return matrix_transpose(matrix_lower_triangular(matrix));
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_lower_triangular(Square<Value, DIMS> const& matrix)
    {
        if constexpr (DIMS == 1) {
            return matrix;
        }

        if constexpr (DIMS > 1) {
            Square<Value, DIMS> result;
            for (Size i{}; i < DIMS; ++i) {
                for (Size j{}; j <= i; ++j) {
                    Value sum{};
                    for (Size k{}; k < j; ++k) {
                        if (j == i) {
                            sum += std::pow(result[j, k], 2);
                            result[j, j] = std::sqrt(matrix[j, j] - sum);
                        } else {
                            sum += (result[i, k] * result[j, k]);
                            result[i, j] = (matrix[i, j] - sum) / result[j, j];
                        }
                    }
                }
            }
            return result;
        }
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_sum(Matrix<Value, ROWS, COLS> const& left,
                                                       Matrix<Value, ROWS, COLS> const& right) noexcept
    {
        Matrix<Value, ROWS, COLS> result;
        for (Size i{}; i < ROWS; ++i) {
            for (Size j{}; j < COLS; ++j) {
                result[i, j] = left[i, j] + right[i, j];
            }
        }
        return result;
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_difference(Matrix<Value, ROWS, COLS> const& left,
                                                              Matrix<Value, ROWS, COLS> const& right) noexcept
    {
        Matrix<Value, ROWS, COLS> result;
        for (Size i{}; i < ROWS; ++i) {
            for (Size j{}; j < COLS; ++j) {
                result[i, j] = left[i, j] - right[i, j];
            }
        }
        return result;
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_scale(Matrix<Value, ROWS, COLS> const& matrix, Value const scale)
    {
        if (scale == std::numeric_limits<Value>::max()) {
            throw Error{"Multiplication by inf!\n"};
        }

        Matrix<Value, ROWS, COLS> result;
        for (Size i{}; i < ROWS; ++i) {
            for (Size j{}; j < COLS; ++j) {
                result[i, j] = matrix[i, j] * scale;
            }
        }
        return result;
    }

    template <Arithmetic Value, Size LEFT_ROWS, Size LEFT_COLS_RIGHT_ROWS, Size RIGHT_COLS>
    [[nodiscard]] Matrix<Value, LEFT_ROWS, RIGHT_COLS>
    matrix_product(Matrix<Value, LEFT_ROWS, LEFT_COLS_RIGHT_ROWS> const& left,
                   Matrix<Value, LEFT_COLS_RIGHT_ROWS, RIGHT_COLS> const& right) noexcept
    {
        Matrix<Value, LEFT_ROWS, RIGHT_COLS> result;
        for (Size i{}; i < left.rows(); ++i) {
            for (Size j{}; j < right.cols(); ++j) {
                Value sum{};
                for (Size k{}; k < left.cols(); ++k) {
                    sum += left[i, k] * right[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> matrix_power(Square<Value, DIMS> const& matrix, Value const power) noexcept
    {
        if (power == 1) {
            return matrix;
        }

        Square<Value, DIMS> result{matrix};
        for (Size i{}; i < power; ++i) {
            result = matrix_product(result, matrix);
        }
        return result;
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> operator+(Matrix<Value, ROWS, COLS> const& left,
                                                      Matrix<Value, ROWS, COLS> const& right) noexcept
    {
        return matrix_sum(left, right);
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> operator-(Matrix<Value, ROWS, COLS> const& left,
                                                      Matrix<Value, ROWS, COLS> const& right) noexcept
    {
        return matrix_difference(left, right);
    }

    template <Arithmetic Value, Size LEFT_ROWS, Size LEFT_COLS_RIGHT_ROWS, Size RIGHT_COLS>
    [[nodiscard]] Matrix<Value, LEFT_ROWS, RIGHT_COLS>
    operator*(Matrix<Value, LEFT_ROWS, LEFT_COLS_RIGHT_ROWS> const& left,
              Matrix<Value, LEFT_COLS_RIGHT_ROWS, RIGHT_COLS> const& right) noexcept
    {
        return matrix_product(left, right);
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> operator*(Value const scale, Matrix<Value, ROWS, COLS> const& matrix)
    {
        try {
            return matrix_scale(matrix, scale);
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> operator*(Matrix<Value, ROWS, COLS> const& matrix, Value const scale)
    {
        try {
            return matrix_scale(matrix, scale);
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size ROWS, Size COLS>
    [[nodiscard]] Matrix<Value, ROWS, COLS> operator/(Matrix<Value, ROWS, COLS> const& matrix, Value const scale)
    {
        try {
            return matrix_scale(matrix, 1 / scale);
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> operator/(Square<Value, DIMS> const& left, Square<Value, DIMS> const& right)
    {
        try {
            return matrix_product(left, matrix_inverse(right));
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value, Size DIMS>
    [[nodiscard]] Square<Value, DIMS> operator^(Square<Value, DIMS> const& matrix, Value const power)
    {
        return matrix_power(matrix, power);
    }

}; // namespace Linalg::Stack

#endif // STACK_VECTOR_HPP