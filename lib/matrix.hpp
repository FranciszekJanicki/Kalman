#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "arithmetic.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <expected>
#include <fmt/core.h>
#include <initializer_list>
#include <utility>
#include <vector>

/* OVERVIEW:
    -create matrixes of given sizes using Matrix::Matrix(...) constructors (round bracket
    initialization) or using Matrix::ones(...), Matrix::zeros(...) and Matrix::eye(...)  factory
    functions

    -create matrixes of given data using Matrix::Matrix{...} constructors (curly bracket initialization)
    or using Matrix::Matrix(...), Matrix::row(...), Matrix::column(...) factory functions
    (overloads with std::initializer_list, be careful, because {} init will always call these overloads)

    -create row, column vectors  and diagonal matrixes with given data using Matrix::Matrix(tag, ...)
    constructors (round bracket initialiation, first param being tag) or using Matrix::row(...),
    Matrix::column(...) and Matrix::diagonal(...) factory functions

    -assign data using operator= assingment operators or using Matrix::data(...) member functions
    -access data using Matrix() conversion operators or using Matrix::data() member functions

    -transpose using Matrix::transpose() and invert using Matrix::inver() member functions (invertion
    will fail if not possible)

    -multiply matrixwith matrix, multiply scalar with matrixand matrixwith scalar,
    divide matrixwith Matrix(same as multiplying by inverse), add matrixes, substract matrixes, of course
    all if dimensions are correct for each of these operations

    -you can printf matrixusing Matrix::print() member function (using printf(), can change to std::print
    if compiler supports it)

    -full interface for data structure (std::vector<std::vector<type>> here)

    -full constexpr support (remember that dynamic memory allocated at compile time, stays at compile time- if you want
    to perform some matrixcalculations at compile time and then get result to run time, use
    std::array<std::array<type>> and copy from data (Matrix::data() accessors) to array, using
    Matrix::rows() and Matrix::cols() to specify std::arrays dimensions)
*/

namespace Linalg {

    template <Arithmetic Value>
    struct Matrix {
    public:
        enum struct Error {
            WRONG_DIMS,
            SINGULARITY,
            BAD_ALLOC,
            BAD_ACCESS,
        };

        using Size = std::size_t;
        using VectorData = std::vector<Value>;
        using MatrixData = std::vector<std::vector<Value>>;
        using ExpectedMatrix = std::expected<Matrix, Error>;
        using ExpectedDet = std::expected<Value, Error>;
        using Unexpected = std::unexpected<Error>;

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

        [[nodiscard]] static constexpr ExpectedMatrix
        minor(Matrix const& matrix, Size const row, Size const column, Size const dimensions)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }
            // assert cofactor isnt calculated for minor bigger than matrix
            assert(dimensions <= row && dimensions <= column);
            if (dimensions > row || dimensions > column) {
                return Unexpected{Error::WRONG_DIMS};
            }
            // minor is scalar, can omit later code
            if (dimensions == 0) {
                return ExpectedMatrix{matrix};
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
            return ExpectedMatrix{std::move(minor)};
        }

        [[nodiscard]] static constexpr ExpectedDet determinant(Matrix const& matrix, Size const dimensions)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // assert minor isnt bigger than matrix
            assert(rows >= dimensions && columns >= dimensions);
            if (rows < dimensions || columns < dimensions) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedDet{matrix[0][0]};
            }
            // matrix is 2x2 matrix, can omit later code
            if (dimensions == 2) {
                return ExpectedDet{(matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1])};
            }

            auto det{static_cast<Value>(0)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size column{0}; column < dimensions; ++column) {
                // cofactor of matrix[0][column]

                if (auto result_minor{Matrix::minor(matrix, 0, column, dimensions)}; result_minor.has_value()) {
                    minor = std::move(result_minor).value();
                } else {
                    print(result_minor.error());
                    std::unreachable();
                }

                if (auto result_det{Matrix::determinant(minor, dimensions - 1)}; result_det.has_value()) {
                    det += sign * matrix[0][column] * result_det.value();
                } else {
                    print(result_det.error());
                    std::unreachable();
                }

                // alternate sign
                sign *= static_cast<Value>(-1);
            }

            return ExpectedDet{det};
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

        [[nodiscard]] static constexpr ExpectedMatrix adjoint(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedMatrix{matrix};
            }

            auto complement{Matrix::make_zeros(dimensions, dimensions)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size row{0}; row < dimensions; ++row) {
                for (Size column{0}; column < dimensions; column++) {
                    // get cofactor of matrix[row][column]

                    if (auto result_minor{Matrix::minor(matrix, row, column, dimensions)}; result_minor.has_value()) {
                        minor = std::move(result_minor).value();
                    } else {
                        print(result_minor.error());
                        std::unreachable();
                    }

                    // sign of adj[column][row] positive if sum of row and column indexes is even
                    if ((row + column) % 2 == 0) {
                        sign = static_cast<Value>(1);
                    } else {
                        sign = static_cast<Value>(-1);
                    }

                    // complement is matrixof determinants of minors with alternating signs!!!
                    if (auto result_det{determinant(minor, dimensions - 1)}; result_det.has_value()) {
                        complement[row][column] = sign * result_det.value();
                    } else {
                        print(result_det.error());
                        std::unreachable();
                    }
                }
            }

            // adjoSize is transposed of complement matrix
            return ExpectedMatrix{Matrix::transpose(complement)};
        }

        [[nodiscard]] static constexpr ExpectedMatrix inverse(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedMatrix{matrix};
            }

            if (auto result_det{Matrix::determinant(matrix, dimensions)}; result_det.has_value()) {
                auto const det{std::move(result_det).value()};

                // assert correct determinant
                assert(det != 0);
                if (det == 0) {
                    return Unexpected{Error::SINGULARITY};
                }

                if (auto result_adjoint{Matrix::adjoint(matrix)}; result_adjoint.has_value()) {
                    auto const adjoint{std::move(result_adjoint).value()};

                    // inverse is adjoint matrixdivided by det factor
                    // division is multiplication by inverse
                    return ExpectedMatrix{Matrix::scale(adjoint, 1 / det)};
                } else {
                    print(result_adjoint.error());
                    std::unreachable();
                }
            } else {
                print(result_det.error());
                std::unreachable();
            }
        }

        [[nodiscard]] static constexpr ExpectedMatrix upper_triangular(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // matrix is scalar
            if (dimensions == 1)
                return ExpectedMatrix{matrix};

            // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
            return ExpectedMatrix{Matrix::transpose(Matrix::lower_triangular(matrix))};
        }

        [[nodiscard]] static constexpr ExpectedMatrix lower_triangular(Matrix const& matrix)
        {
            Size const rows{matrix.rows()};
            Size const columns{matrix.columns()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{Error::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions = rows; // = columns;
            // matrix is scalar
            if (dimensions == 1) {
                return ExpectedMatrix{matrix};
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
            return ExpectedMatrix{std::move(lower_triangular)};
        }

        [[nodiscard]] static constexpr ExpectedMatrix multiply(Matrix const& left, Matrix const& right)
        {
            Size const left_rows{left.rows()};
            Size const right_rows{right.rows()};
            Size const left_columns = {left.columns()};
            Size const right_columns{right.columns()};

            // assert correct dimensions
            assert(left_columns == right_rows);
            if (left_columns != right_rows) {
                return Unexpected{Error::WRONG_DIMS};
            }

            Size const multiply_rows{left_rows};
            Size const multiply_columns{right_columns};
            auto multiply{Matrix::make_zeros(multiply_rows, multiply_columns)};

            for (Size left_row{0}; left_row < left_rows; ++left_row) {
                for (Size right_column{0}; right_column < right_columns; ++right_column) {
                    Value sum{0};
                    for (Size left_column{0}; left_column < left_columns; ++left_column) {
                        sum += left[left_row][left_column] * right[left_column][right_column];
                    }
                    multiply[left_row][right_column] = sum;
                }
            }
            return ExpectedMatrix{std::move(multiply)};
        }

        [[nodiscard]] static constexpr Matrix add(Matrix const& left, Matrix const& right) noexcept
        {
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            auto sum{Matrix::make_zeros(left.rows(), left.columns())};
            for (Size row{0}; row < left.rows(); ++row) {
                for (Size column{0}; column < left.columns(); ++column) {
                    sum[row][column] = left[row][column] + right[row][column];
                }
            }
            return sum;
        }

        [[nodiscard]] static constexpr Matrix substract(Matrix const& left, Matrix const& right) noexcept
        {
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

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

        constexpr Matrix& operator+=(this Matrix& self, Matrix const& other) noexcept
        {
            // assert correct dimensions
            assert(other.rows() == self.rows());
            assert(other.columns() == self.rows());

            for (Size row{0}; row < self.rows(); ++row) {
                for (Size column{0}; column < self.columns(); ++column) {
                    self[row][column] += other[row][column];
                }
            }
            return self;
        }

        constexpr Matrix& operator-=(this Matrix& self, Matrix const& other) noexcept
        {
            // assert correct dimensions
            assert(other.rows() == self.rows());
            assert(other.columns() == self.rows());

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
            // assert correct dimensions
            assert(self.columns() == other.rows());

            if (auto multiply{Matrix::multiply(self, other)}; multiply.has_value()) {
                self = std::move(multiply).value();
                return self;
            } else {
                print(multiply.error());
                std::unreachable();
            }
        }

        constexpr Matrix& operator/=(this Matrix& self, Value const& factor)
        {
            // assert no division by 0!!!
            assert(factor != 0);

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
            // assert correct dimensions
            assert(self.columns() == other.rows());

            // division is multiplication by inverse
            if (auto inverse{Matrix::inverse(other)}; inverse.has_value()) {
                if (auto multiply{Matrix::multiply(self, inverse.value())}; multiply.has_value()) {
                    self = std::move(multiply).value();
                    return self;
                } else {
                    print(multiply.error());
                    std::unreachable();
                }
            } else {
                print(inverse.error());
                std::unreachable();
            }
        }

        constexpr Matrix& operator^=(this Matrix& self, Value const& factor)
        {
            assert(self.is_square());
            for (Value i{}; i < factor - 1; ++i) {
                self *= self;
            }
            return self;
        }

        friend constexpr Matrix operator+(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            return Matrix::add(left, right);
        }

        friend constexpr Matrix operator-(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

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
            // assert correct dimensions
            assert(left.columns() == right.rows());

            if (auto multiply{Matrix::multiply(left, right)}; multiply.has_value()) {
                // i wouldnt get RVO anyway, sinve .value() isnt a local identifier, its a part of a local identifier,
                // multiply is a local identifier, but im retutning its value
                return std::move(multiply).value();
            } else {
                print(multiply.error());
                std::unreachable();
            }
        }

        friend constexpr Matrix operator/(Matrix const& matrix, Value const& factor)
        {
            // assert no division by 0!!!
            assert(factor != 0);

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            // division is multiplication by inverse
            return Matrix::scale(matrix, 1 / factor);
        }

        friend constexpr Matrix operator/(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.columns() == right.rows());

            // division is multiplication by inverse
            if (auto inverse{Matrix::inverse(right)}; inverse.has_value()) {
                if (auto multiply{Matrix::multiply(left, inverse.value())}; multiply.has_value()) {
                    return std::move(multiply).value();
                } else {
                    print(multiply.error());
                    std::unreachable();
                }
            } else {
                print(inverse.error());
                std::unreachable();
            }
        }

        friend constexpr Matrix operator^(Matrix const& matrix, Value const& factor)
        {
            assert(matrix.is_square());
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

        [[nodiscard]] constexpr VectorData const& operator[](this Matrix const& self, Size const row) noexcept
        {
            assert(row <= self.rows());
            return self.data_[row];
        }

        [[nodiscard]] constexpr VectorData& operator[](this Matrix& self, Size const row) noexcept
        {
            assert(row <= self.rows());
            return self.data_[row];
        }

        [[nodiscard]] constexpr Value& operator[](this Matrix& self, Size const row, Size const column) noexcept
        {
            assert(row <= self.rows());
            assert(column <= self.columns());
            return self.data_[row][column];
        }

        [[nodiscard]] constexpr Value const&
        operator[](this Matrix const& self, Size const row, Size const column) noexcept
        {
            assert(row <= self.rows());
            assert(column <= self.columns());
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
            assert(new_row.size() == self.columns());
            self.data_.insert(std::next(self.data_.begin(), row), new_row);
        }

        constexpr void insert_column(this Matrix& self, Size const column, VectorData const& new_column)
        {
            assert(new_column.size() == self.rows());
            for (auto const& row : self.data_) {
                row.insert(std::next(row.begin(), column), new_column[column]);
            }
        }

        constexpr void delete_row(this Matrix& self, Size const row)
        {
            assert(row <= self.rows());
            self.data_.erase(std::next(self.data_.begin(), row));
        }

        constexpr void delete_column(this Matrix& self, Size const column)
        {
            assert(column <= self.columns());
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
            if (auto inverse{Matrix::inverse(self)}; inverse.has_value()) {
                self = std::move(inverse).value();
            } else {
                print(inverse.error());
                std::unreachable();
            }
        }

    private:
        static constexpr const char* error_to_string(Error const Error) noexcept
        {
            switch (Error) {
                case Error::WRONG_DIMS:
                    return "Wrong dims";
                case Error::SINGULARITY:
                    return "Singularity";
                case Error::BAD_ALLOC:
                    return "Bad alloc";
                default:
                    return "None";
            }
        }

        static constexpr void print(Error const Error) noexcept
        {
            fmt::print("{}", error_to_string(Error));
        }

        MatrixData data_{};
    };
}; // namespace Linalg

#endif // MATRIX_HPP