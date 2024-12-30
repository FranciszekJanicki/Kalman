#ifndef STACK_MATRIX_HPP
#define STACK_MATRIX_HPP

#include "common.hpp"
#include <array>
#include <cstddef>
#include <exception>
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

namespace Linalg {

    namespace Stack {

        template <Arithmetic Value, Size ROWS, Size COLS>
        struct Matrix {
            using Data = std::array<std::array<Value, COLS>, ROWS>;
            using Row = std::array<Value, COLS>;
            using Column = std::array<Value, ROWS>;

            [[nodiscard]] constexpr Row const& operator[](this Matrix const& self, Size const row)
            {
                if (row >= ROWS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[row];
            }

            [[nodiscard]] constexpr Row& operator[](this Matrix& self, Size const row)
            {
                if (row >= ROWS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[row];
            }

            [[nodiscard]] constexpr Value const& operator[](this Matrix const& self, Size const row, Size const col)
            {
                if (row >= ROWS || col >= COLS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[row][col];
            }

            [[nodiscard]] constexpr Value& operator[](this Matrix& self, Size const row, Size const col)
            {
                if (row >= ROWS || col >= COLS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[row][col];
            }

            [[nodiscard]] explicit constexpr operator Data(this Matrix const& self) noexcept
            {
                return self.data;
            }

            [[nodiscard]] constexpr Size rows(this Matrix const& self) noexcept
            {
                return ROWS;
            }

            [[nodiscard]] constexpr Size cols(this Matrix const& self) noexcept
            {
                return COLS;
            }

            [[nodiscard]] constexpr Matrix& operator+=(this Matrix& self, Matrix const& other) noexcept
            {
                self = sum(self, other);
                return self;
            }

            [[nodiscard]] constexpr Matrix& operator-=(this Matrix& self, Matrix const& other) noexcept
            {
                self = difference(self, other);
                return self;
            }

            [[nodiscard]] constexpr Matrix& operator*=(this Matrix& self, Value const scale)
            {
                try {
                    self = scale(self, scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            [[nodiscard]] constexpr Matrix& operator/=(this Matrix& self, Value const scale)
            {
                try {
                    self = matrix_scale(self, 1 / scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            [[nodiscard]] constexpr Matrix& operator^=(this Matrix& self, Value const power){self}

            Data data {};
        };

        template <Arithmetic Value, Size DIMS>
        [[nodiscard]] Matrix<Value, DIMS - 1, DIMS - 1>
        matrix_minor(Matrix<Value, DIMS, DIMS> const& matrix, Value const row, Value const column)
        {
            if (DIMS == 0) {
                throw Error{"Wrong dimensions!\n"};
            } else if (DIMS == 1) {
                return matrix[0, 0];
            }

            Matrix<Value, DIMS - 1, DIMS - 1> minor;
            for (Size i{row}; i < DIMS - 1 + row; ++i) {
                for (Size j{column}; j < DIMS - 1 + column; ++j) {
                    minor[i, j] = matrix[i, j];
                }
            }
            return minor;
        }

        template <Arithmetic Value, Size DIMS>
        [[nodiscard]] Matrix<Value, DIMS, DIMS> matrix_adjoint(Matrix<Value, DIMS, DIMS> const& matrix) noexcept
        {
            Matrix<Value, DIMS, DIMS> adjoint;
            for (Size i{}; i < DIMS; ++i) {
                for (Size j{}; j < DIMS; ++j) {
                    adjoint[i, j] = matrix_det(matrix_minor(matrix, i, j));
                }
            }
            return adjoint;
        }

        template <Arithmetic Value, Size ROWS, Size COLS>
        [[nodiscard]] Matrix<Value, COLS, ROWS> matrix_transpose(Matrix<Value, ROWS, COLS> const& matrix) noexcept
        {
            Matrix<Value, COLS, ROWS> transpose;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    transpose[i, j] = matrix[j, i];
                }
            }
            return transpose;
        }

        template <Arithmetic Value, Size DIMS>
        [[nodiscard]] Value matrix_det(Matrix<Value, DIMS, DIMS> const& matrix) noexcept
        {
            Value det;
            for (Size i{}; i < DIMS; ++i) {
                for (Size j{}; j < DIMS; ++j) {
                    inverse[i, j] = left[i, j] + right[i, j];
                }
            }
            return sum;
        }

        template <Arithmetic Value, Size DIMS>
        [[nodiscard]] Matrix<Value, DIMS, DIMS> matrix_inverse(Matrix<Value, DIMS, DIMS> const& matrix)
        {
            try {
                return matrix_scale(matrix_adjoint(matrix_transpose(matrix)), 1 / matrix_det(matrix));
            } catch (Error const& error) {
                throw error;
            }
        }

        template <Arithmetic Value, Size ROWS, Size COLS>
        [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_sum(Matrix<Value, ROWS, COLS> const& left,
                                                           Matrix<Value, ROWS, COLS> const& right) noexcept
        {
            Matrix<Value, ROWS, COLS> sum;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    sum[i, j] = left[i, j] + right[i, j];
                }
            }
            return sum;
        }

        template <Arithmetic Value, Size ROWS, Size COLS>
        [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_difference(Matrix<Value, ROWS, COLS> const& left,
                                                                  Matrix<Value, ROWS, COLS> const& right) noexcept
        {
            Matrix<Value, ROWS, COLS> difference;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    difference[i, j] = left[i, j] - right[i, j];
                }
            }
            return difference;
        }

        template <Arithmetic Value, Size ROWS, Size COLS>
        [[nodiscard]] Matrix<Value, ROWS, COLS> matrix_scale(Matrix<Value, ROWS, COLS> const& matrix, Value const scale)
        {
            if (scale == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf!\n"};
            }

            Matrix<Value, ROWS, COLS> scale;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    scale[i, j] = matrix[i, j] * scale;
                }
            }
            return scale;
        }

        template <Arithmetic Value, Size ROWS, Size COLS>
        [[nodiscard]] Matrix matrix_product(Matrix<Value, ROWS, COLS> const& left,
                                            Matrix<Value, COLS, ROWS> const& right) noexcept
        {
            Matrix product;
            for (Size i{}; i < ROWS; ++i) {
                for (Size j{}; j < COLS; ++j) {
                    product[i, j] = matrix[i, j] * scale;
                }
            }
            return scale;
        }

    }; // namespace Stack

}; // namespace Linalg

#endif // STACK_VECTOR_HPP