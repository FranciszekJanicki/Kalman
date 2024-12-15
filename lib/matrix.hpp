#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "arithmetic.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <expected>
#include <fmt/core.h>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Linalg {

    template <Arithmetic Value>
    struct Matrix {
    public:
        using Size = std::size_t;
        using VectorData = std::vector<Value>;
        using MatrixData = std::vector<std::vector<Value>>;

        [[nodiscard]] static constexpr Matrix
        make_matrix(std::initializer_list<const std::initializer_list<Value>> const data)
        {
            Matrix matrix{};
            matrix.data_.reserve(data.size());
            for (auto const& row : data) {
                auto& column{matrix.data_.emplace_back()};
                column.reserve(row.size());
                for (auto const& col : row) {
                    column.push_back(col);
                }
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_zeros(Size const rows, Size const columns)
        {
            Matrix matrix{};
            matrix.data_.reserve(rows);
            for (Size row{0}; row < rows; ++row) {
                auto& column{matrix.data_.emplace_back()};
                column.reserve(columns);
                for (Size col{0}; col < columns; ++col) {
                    column.emplace_back();
                }
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_ones(Size const rows, Size const columns)
        {
            Matrix matrix{};
            matrix.data_.reserve(rows);
            for (Size row{0}; row < rows; ++row) {
                auto& column{matrix.data_.emplace_back()};
                column.reserve(columns);
                for (Size col{0}; col < columns; ++col) {
                    column.push_back(Value{1});
                }
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_diagonal(std::initializer_list<Value> const diagonal)
        {
            Matrix matrix{};
            matrix.data_.reserve(diagonal.size());
            for (Size row{0}; row < diagonal.size(); ++row) {
                auto& column{matrix.data_.emplace_back()};
                column.reserve(diagonal.size());
                for (Size col{0}; col < diagonal.size(); ++col) {
                    if (col == row) {
                        column.push_back(diagonal.begin() + col);
                    } else {
                        column.emplace_back();
                    }
                }
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_eye(Size const dimensions)
        {
            Matrix matrix{};
            matrix.data_.reserve(dimensions);
            for (Size row{0}; row < dimensions; ++row) {
                auto& column{matrix.data_.emplace_back()};
                column.reserve(dimensions);
                for (Size col{0}; col < dimensions; ++col) {
                    if (col == row) {
                        column.push_back(Value{1});
                    } else {
                        column.emplace_back();
                    }
                }
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_row(Size const rows)
        {
            Matrix matrix{};
            matrix.data_.reserve(rows);
            for (Size row{}; row < rows; ++row) {
                matrix.data_.emplace_back();
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_row(std::initializer_list<Value> const data)
        {
            Matrix matrix{};
            auto const columns{data.size()};
            matrix.data_.reserve(columns);
            for (auto const col : data) {
                matrix.data_.push_back(col);
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_column(Size const columns)
        {
            Matrix matrix{};
            auto& column{matrix.data_.emplace_back()};
            column.reserve(columns);
            for (Size col{}; col < columns; ++col) {
                column.emplace_back();
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix make_column(std::initializer_list<Value> const data)
        {
            Matrix matrix{};
            auto& column{matrix.emplace_back()};
            column.reserve(data.size());
            for (auto const row : data) {
                matrix.push_back(row);
            }
            return matrix;
        }

        [[nodiscard]] static constexpr Matrix
        minor(Matrix const& matrix, Size const row, Size const column, Size const dimensions)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            assert(rows == columns);
            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            // assert cofactor isnt calculated for minor bigger than matrix
            assert(dimensions <= row && dimensions <= column);
            if (dimensions > row || dimensions > column) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            // minor is scalar, can omit later code
            if (dimensions == 0) {
                return matrix;
            }

            auto minor{make_zeros(dimensions, dimensions)};
            Size cof_row{0};
            Size cof_column{0};
            for (Size row_{0}; row_ < dimensions; ++row_) {
                for (Size column_{0}; column_ < dimensions; ++column_) {
                    // copying into cofactor matrixonly those element which are not in given row and column
                    if (row_ != row && column_ != column) {
                        minor[cof_row][cof_column++] = matrix[row_][column_];

                        // row is filled, so increase row index and reset column index
                        if (cof_column == dimensions - 1) {
                            cof_column = 0;
                            ++cof_row;
                        }
                    }
                }
            }
            return minor;
        }

        [[nodiscard]] static constexpr Value determinant(Matrix const& matrix, Size const dimensions)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            assert(rows == columns);
            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // assert minor isnt bigger than matrix
            assert(rows >= dimensions && columns >= dimensions);
            if (rows < dimensions || columns < dimensions) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return matrix[0][0];
            }
            // matrix is 2x2 matrix, can omit later code
            if (dimensions == 2) {
                return (matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]);
            }

            auto det{static_cast<Value>(0)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size column{0}; column < dimensions; ++column) {
                // cofactor of matrix[0][column]

                try {
                    minor = Matrix::minor(matrix, 0, column, dimensions);
                    try {
                        det += sign * matrix[0][column] * Matrix::determinant(minor, dimensions - 1);
                    } catch (std::runtime_error const& error) {
                        throw error;
                    }
                } catch (std::runtime_error const& error) {
                    throw error;
                }

                // alternate sign
                sign *= static_cast<Value>(-1);
            }

            return det;
        }

        [[nodiscard]] static constexpr Matrix transpose(Matrix const& matrix)
        {
            Size const new_rows{matrix.rows()};
            Size const new_columns{matrix.columns()};

            // matrix is scalar, can omit later code
            if ((new_rows == new_columns) == 1)
                return matrix;

            auto transpose{Matrix::make_zeros(new_rows, new_columns)};
            for (Size row{0}; row < new_rows; ++row) {
                for (Size column{0}; column < new_columns; ++column) {
                    transpose[row][column] = matrix[column][row];
                }
            }
            return transpose;
        }

        [[nodiscard]] static constexpr Matrix adjoint(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            assert(rows == columns);
            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return matrix;
            }

            auto complement{Matrix::make_zeros(dimensions, dimensions)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size row{0}; row < dimensions; ++row) {
                for (Size column{0}; column < dimensions; column++) {
                    // get cofactor of matrix[row][column]

                    try {
                        minor = Matrix::minor(matrix, row, column, dimensions);
                    } catch (std::runtime_error const& error) {
                        throw error;
                    }

                    // sign of adj[column][row] positive if sum of row and column indexes is even
                    if ((row + column) % 2 == 0) {
                        sign = static_cast<Value>(1);
                    } else {
                        sign = static_cast<Value>(-1);
                    }

                    // complement is matrixof determinants of minors with alternating signs!!!
                    try {
                        complement[row][column] = sign * Matrix::determinant(minor, dimensions - 1);
                    } catch (std::runtime_error const& error) {
                        throw error;
                    }
                }
            }

            // adjoSize is transposed of complement matrix
            return Matrix::transpose(complement);
        }

        [[nodiscard]] static constexpr Matrix inverse(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return matrix;
            }

            try {
                auto det{Matrix::determinant(matrix, dimensions)};

                // assert correct determinant
                if (det == 0) {
                    throw std::runtime_error{"Singularity\n"};
                }

                try {
                    // inverse is adjoint matrixdivided by det factor
                    // division is multiplication by inverse
                    return Matrix::scale(Matrix::adjoint(matrix), 1 / det);
                } catch (std::runtime_error const& error) {
                    throw error;
                }

            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        [[nodiscard]] static constexpr Matrix upper_triangular(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            assert(rows == columns);
            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar
            if (dimensions == 1)
                return matrix;

            // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
            try {
                return Matrix::transpose(Matrix::lower_triangular(matrix));
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        [[nodiscard]] static constexpr Matrix lower_triangular(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            assert(rows == columns);
            if (rows != columns) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // matrixsquare
            Size const dimensions = rows; // = columns;
            // matrix is scalar
            if (dimensions == 1) {
                return matrix;
            }

            auto lower_triangular{Matrix::make_zeros(dimensions, dimensions)};

            // decomposing matrix matrixinto lower triangular
            for (Size row{0}; row < dimensions; ++row) {
                for (Size column{0}; column <= row; ++column) {
                    Value sum{};

                    // summation for diagonals
                    if (column == row) {
                        for (Size sum_col{0}; sum_col < column; ++sum_col) {
                            sum += std::pow(lower_triangular[column][sum_col], 2);
                        }
                        lower_triangular[column][column] = std::sqrt(matrix[column][column] - sum);
                    } else {
                        // evaluating L(row, column) using L(column, column)
                        for (Size sum_col{0}; sum_col < column; ++sum_col) {
                            sum += (lower_triangular[row][sum_col] * lower_triangular[column][sum_col]);
                        }
                        lower_triangular[row][column] = (matrix[row][column] - sum) / lower_triangular[column][column];
                    }
                }
            }
            return lower_triangular;
        }

        [[nodiscard]] static constexpr Matrix product(Matrix const& left, Matrix const& right)
        {
            Size const left_rows{left.rows()};
            Size const right_rows{right.rows()};
            Size const left_columns = {left.columns()};
            Size const right_columns{right.columns()};

            assert(left_columns == right_rows);
            if (left_columns != right_rows) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            Size const product_rows{left_rows};
            Size const product_columns{right_columns};
            auto product{Matrix::make_zeros(product_rows, product_columns)};

            for (Size left_row{0}; left_row < left_rows; ++left_row) {
                for (Size right_column{0}; right_column < right_columns; ++right_column) {
                    Value sum{0};
                    for (Size left_column{0}; left_column < left_columns; ++left_column) {
                        sum += left[left_row][left_column] * right[left_column][right_column];
                    }
                    product[left_row][right_column] = sum;
                }
            }
            return product;
        }

        [[nodiscard]] static constexpr Matrix sum(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.columns() || left.rows() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            auto sum{Matrix::make_zeros(left.rows(), left.columns())};
            for (Size row{0}; row < left.rows(); ++row) {
                for (Size column{0}; column < left.columns(); ++column) {
                    sum[row][column] = left[row][column] + right[row][column];
                }
            }
            return sum;
        }

        [[nodiscard]] static constexpr Matrix substract(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.columns() || left.rows() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            auto difference{Matrix::make_zeros(left.rows(), left.columns())};
            for (Size row{0}; row < left.rows(); ++row) {
                for (Size column{0}; column < left.columns(); ++column) {
                    difference[row][column] = left[row][column] - right[row][column];
                }
            }
            return difference;
        }

        [[nodiscard]] static constexpr Matrix scale(Matrix const& matrix, Value const factor)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }
            // factor is 0 then return matrixof zeros
            else if (factor == 0) {
                return Matrix::make_zeros(matrix.rows(), matrix.columns());
            }

            auto scale{Matrix::make_zeros(rows, columns)};
            for (Size row{0}; row < rows; ++row) {
                for (Size column{0}; column < columns; ++column) {
                    scale[row][column] = matrix[row][column] * factor;
                }
            }
            return scale;
        }

        constexpr Matrix() noexcept = default;

        explicit constexpr Matrix(std::initializer_list<std::initializer_list<Value> const> const data)
        {
            *this = make_matrix(data);
        }

        constexpr Matrix(Size const rows, Size const columns)
        {
            *this = make_zeros(rows, columns);
        }

        explicit constexpr Matrix(MatrixData&& data) noexcept : data_{std::forward<MatrixData>(data)}
        {}

        explicit constexpr Matrix(MatrixData const& data) : data_{data}
        {}

        constexpr Matrix(Matrix const& other) = default;

        constexpr Matrix(Matrix&& other) noexcept = default;

        constexpr ~Matrix() noexcept = default;

        constexpr Matrix& operator=(Matrix const& other) = default;

        constexpr Matrix& operator=(Matrix&& other) noexcept = default;

        constexpr void operator=(this Matrix& self, MatrixData&& data) noexcept
        {
            self.data_ = std::forward<MatrixData>(data);
        }

        constexpr void operator=(this Matrix& self, MatrixData const& data)
        {
            self.data_ = data;
        }

        constexpr Matrix& operator+=(this Matrix& self, Matrix const& other)
        {
            if (self.columns() != other.columns() || self.rows() != other.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            for (Size row{0}; row < self.rows(); ++row) {
                for (Size column{0}; column < self.columns(); ++column) {
                    self[row][column] += other[row][column];
                }
            }
            return self;
        }

        constexpr Matrix& operator-=(this Matrix& self, Matrix const& other)
        {
            if (self.columns() != other.columns() || self.rows() != other.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            for (Size row{0}; row < self.rows(); ++row) {
                for (Size column{0}; column < self.columns(); ++column) {
                    self[row][column] -= other[row][column];
                }
            }
            return self;
        }

        constexpr Matrix& operator*=(this Matrix& self, Value const& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return self;
            }

            self = scale(self, factor);
            return self;
        }

        constexpr Matrix& operator*=(this Matrix& self, Matrix const& other)
        {
            if (self.columns() != other.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            try {
                self = Matrix::product(self, other);
                return self;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        constexpr Matrix& operator/=(this Matrix& self, Value const& factor)
        {
            // assert no division by 0!!!
            if (factor == 0) {
                throw std::runtime_error{"Division by 0\n"};
            }

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return self;
            }

            // division is multiplication by inverse
            self = scale(self, 1 / factor);
            return self;
        }

        constexpr Matrix& operator/=(this Matrix& self, Matrix const& other)
        {
            if (self.columns() != other.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // division is multiplication by inverse
            try {
                auto inverse{Matrix::inverse(other)};
                try {
                    self = Matrix::product(self, inverse);
                    return self;
                } catch (std::runtime_error const& error) {
                    throw error;
                }
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        constexpr Matrix& operator^=(this Matrix& self, Value const& factor)
        {
            if (!self.is_square()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            for (Value i{}; i < factor - 1; ++i) {
                self *= self;
            }
            return self;
        }

        friend constexpr Matrix operator+(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.columns() || left.rows() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            return Matrix::sum(left, right);
        }

        friend constexpr Matrix operator-(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.columns() || left.rows() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            return Matrix::substract(left, right);
        }

        friend constexpr Matrix operator*(Value const& factor, Matrix const& matrix)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix::scale(matrix, factor);
        }

        friend constexpr Matrix operator*(Matrix const& matrix, Value const& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix::scale(matrix, factor);
        }

        friend constexpr Matrix operator*(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            try {
                auto product{Matrix::product(left, right)};
                return product;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        friend constexpr Matrix operator/(Matrix const& matrix, Value const& factor)
        {
            // assert no division by 0!!!
            if (factor == 0) {
                throw std::runtime_error{"Division by zero\n"};
            }

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            // division is multiplication by inverse
            return Matrix::scale(matrix, 1 / factor);
        }

        friend constexpr Matrix operator/(Matrix const& left, Matrix const& right)
        {
            if (left.columns() != right.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }

            // division is multiplication by inverse
            try {
                auto inverse{Matrix::inverse(right)};
                try {
                    return Matrix::product(left, inverse);
                } catch (std::runtime_error const& error) {
                    throw error;
                }
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        friend constexpr Matrix operator^(Matrix const& matrix, Value const& factor)
        {
            if (!matrix.is_square()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            Matrix result{matrix};
            for (Value i{}; i < factor - 1; ++i) {
                result *= matrix;
            }
            return result;
        }

        explicit constexpr operator MatrixData(this Matrix&& self) noexcept
        {
            return std::forward<Matrix>(self).data_;
        }

        explicit constexpr operator MatrixData(this Matrix const& self) noexcept
        {
            return self.data_;
        }

        [[nodiscard]] constexpr VectorData const& operator[](this Matrix const& self, Size const row)
        {
            if (row > self.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            return self.data_[row];
        }

        [[nodiscard]] constexpr VectorData& operator[](this Matrix& self, Size const row)
        {
            if (row > self.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            return self.data_[row];
        }

        [[nodiscard]] constexpr Value& operator[](this Matrix& self, Size const row, Size const column)
        {
            if (row > self.rows() || column > self.columns()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            return self.data_[row][column];
        }

        [[nodiscard]] constexpr Value const& operator[](this Matrix const& self, Size const row, Size const column)
        {
            if (row > self.rows() || column > self.columns()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            return self.data_[row][column];
        }

        [[nodiscard]] constexpr bool operator<=>(this Matrix const& self, Matrix const& other) noexcept = default;

        constexpr void print(this Matrix const& self) noexcept
        {
            fmt::print("[");
            if (!self.data_.empty()) {
                for (auto& row : self.data_) {
                    fmt::print("[");
                    if (!row.empty()) {
                        for (auto& col : row) {
                            fmt::print("{}", col);
                            if (col != row.back()) {
                                fmt::print(", ");
                            }
                        }
                    }
                    fmt::print("]");
                    if (row != self.data_.back()) {
                        fmt::print(",\n");
                    }
                }
            }
            fmt::print("]\n");
        }

        [[nodiscard]] constexpr MatrixData const& data(this Matrix const& self) noexcept
        {
            return self.data_;
        }

        [[nodiscard]] constexpr MatrixData&& data(this Matrix&& self) noexcept
        {
            return std::forward<Matrix>(self).data_;
        }

        constexpr void data(this Matrix& self, MatrixData&& data) noexcept
        {
            self.data_ = std::forward<MatrixData>(data);
        }

        constexpr void data(this Matrix& self, MatrixData const& data)
        {
            self.data_ = data;
        }

        constexpr void swap(this Matrix& self, Matrix& other)
        {
            std::swap(self.data_, other.data_);
        }

        constexpr void insert_row(this Matrix& self, Size const row, VectorData const& new_row)
        {
            if (new_row.size() != self.columns()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            self.data_.insert(std::next(self.data_.begin(), row), new_row);
        }

        constexpr void insert_column(this Matrix& self, Size const column, VectorData const& new_column)
        {
            if (new_column.size() != self.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            for (auto const& row : self.data_) {
                row.insert(std::next(row.begin(), column), new_column[column]);
            }
        }

        constexpr void delete_row(this Matrix& self, Size const row)
        {
            if (row > self.rows()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            self.data_.erase(std::next(self.data_.begin(), row));
        }

        constexpr void delete_column(this Matrix& self, Size const column)
        {
            if (column > self.columns()) {
                throw std::runtime_error{"Wrong dimensions\n"};
            }
            for (Size row{0}; row < self.rows(); ++row) {
                self.data_[row].erase(std::next(self.data_[row].begin(), column));
            }
        }

        constexpr VectorData const& end_row(this Matrix const& self) noexcept
        {
            return self.data_.back();
        }

        constexpr VectorData& end_row(this Matrix& self) noexcept
        {
            return self.data_.back();
        }

        constexpr VectorData const& begin_row(this Matrix const& self) noexcept
        {
            return self.data_.front();
        }

        constexpr VectorData& begin_row(this Matrix& self) noexcept
        {
            return self.data_.front();
        }

        constexpr VectorData end_column(this Matrix const& self)
        {
            VectorData end_column{};
            end_column.reserve(self.columns());
            for (auto const& row : self.data_) {
                end_column.push_back(row.back());
            }
            return end_column;
        }

        constexpr VectorData begin_column(this Matrix const& self)
        {
            VectorData begin_column{};
            begin_column.reserve(self.columns());
            for (auto const& row : self.data_) {
                begin_column.push_back(row.front());
            }
            return begin_column;
        }

        constexpr void reserve(this Matrix& self, Size const rows, Size const columns)
        {
            self.data_.reserve(rows);
            for (auto& row : self.data_) {
                row.reserve(columns);
            }
        }

        constexpr void resize(this Matrix& self, Size const rows, Size const columns)
        {
            self.data_.resize(rows);
            for (auto& row : self.data_) {
                row.resize(columns);
            }
        }

        constexpr void erase(this Matrix& self)
        {
            self.data_.erase();
        }

        constexpr void clear(this Matrix& self)
        {
            self.data_.clear();
        }

        [[nodiscard]] constexpr bool is_empty(this Matrix const& self) noexcept
        {
            return self.rows() == self.columns() == 0;
        }

        [[nodiscard]] constexpr bool is_square(this Matrix const& self) noexcept
        {
            return self.rows() == self.columns();
        }

        [[nodiscard]] constexpr Size rows(this Matrix const& self) noexcept
        {
            return self.data_.size();
        }

        [[nodiscard]] constexpr Size columns(this Matrix const& self) noexcept
        {
            return self.data_.front().size();
        }

        constexpr VectorData diagonal(this Matrix const& self)
        {
            assert(self.rows() == self.columns());

            VectorData diagonale{};
            diagonale.reserve(self.rows());

            for (Size diag{0}; diag < self.rows(); ++diag) {
                diagonale.push_back(self[diag][diag]);
            }
            return diagonale;
        }

        constexpr void transpose(this Matrix& self)
        {
            self = transpose(self);
        }

        constexpr void invert(this Matrix& self)
        {
            try {
                self = Matrix::inverse(self);
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

    private:
        MatrixData data_{};
    };
}; // namespace Linalg

#endif // MATRIX_HPP