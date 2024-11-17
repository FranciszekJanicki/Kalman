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
        enum struct MatrixError {
            WRONG_DIMS,
            SINGULARITY,
            BAD_ALLOC,
            BAD_ACCESS,
        };

        using Size = std::size_t;
        using VectorData = std::vector<Value>;
        using MatrixData = std::vector<std::vector<Value>>;
        using ExpectedMatrix = std::expected<MatrixData, MatrixError>;
        using ExpectedVector = std::expected<VectorData, MatrixError>;
        using ExpectedDet = std::expected<Value, MatrixError>;
        using Unexpected = std::unexpected<MatrixError>;

        [[nodiscard]] static constexpr Matrix row(std::initializer_list<Value> const row)
        {
            return Matrix{make_row(row)};
        }

        [[nodiscard]] static constexpr Matrix row(Size const rows)
        {
            return Matrix{make_row(rows)};
        }

        [[nodiscard]] static constexpr Matrix column(std::initializer_list<Value> const column)
        {
            return Matrix{make_column(column)};
        }

        [[nodiscard]] static constexpr Matrix column(Size const columns)
        {
            return Matrix{make_column(columns)};
        }

        [[nodiscard]] static constexpr Matrix diagonal(std::initializer_list<Value> const diagonal)
        {
            return Matrix{make_diagonal(diagonal)};
        }

        [[nodiscard]] static constexpr Matrix eye(Size const dimensions)
        {
            return Matrix{make_eye(dimensions)};
        }

        [[nodiscard]] static constexpr Matrix ones(Size const rows, Size const columns)
        {
            return Matrix{make_ones(rows, columns)};
        }

        [[nodiscard]] static constexpr Matrix zeros(Size const rows, Size const columns)
        {
            return Matrix{make_zeros(rows, columns)};
        }

        constexpr Matrix() noexcept = default;

        explicit constexpr Matrix(std::initializer_list<std::initializer_list<Value> const> const data) : data_{data}
        {}

        constexpr Matrix(Size const rows, Size const columns) : data_{make_zeros(rows, columns)}
        {}

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
                    self.data_[row][column] += other.data_[row][column];
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
                    self.data_[row][column] -= other.data_[row][column];
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

            self.data_ = scale(self.data_, factor);
            return self;
        }

        constexpr Matrix& operator*=(this Matrix& self, Matrix const& other)
        {
            // assert correct dimensions
            assert(self.columns() == other.rows());

            if (auto expected_product{Matrix::product(self.data_, other.data_)}; expected_product.has_value()) {
                self.data_ = std::move(expected_product).value();
                return self;
            } else {
                print_error(expected_product.error());
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
            self.data_ = scale(self.data_, 1 / factor);
            return self;
        }

        constexpr Matrix& operator/=(this Matrix& self, Matrix const& other)
        {
            // assert correct dimensions
            assert(self.columns() == other.rows());

            // division is multiplication by inverse
            if (auto expected_inverse{inverse(other.data_)}; expected_inverse.has_value()) {
                if (auto expected_product{product(self.data_, expected_inverse.value())};
                    expected_product.has_value()) {
                    self = std::move(expected_product).value();
                    return self;
                } else {
                    print_error(expected_product.error());
                    std::unreachable();
                }
            } else {
                print_error(expected_inverse.error());
                std::unreachable();
            }
        }

        constexpr Matrix& operator^=(this Matrix& self, Value const& factor)
        {
            assert(self.is_square());
            for (Value i{}; i < factor - 1; ++i) {
                self.data_ *= self.data_;
            }
            return self;
        }

        friend constexpr Matrix operator+(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            return Matrix{sum(left.data_, right.data_)};
        }

        friend constexpr Matrix operator-(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            return Matrix{difference(left.data_, right.data_)};
        }

        friend constexpr Matrix operator*(Value const& factor, Matrix const& matrix)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix{scale(matrix.data_, factor)};
        }

        friend constexpr Matrix operator*(Matrix const& matrix, Value const& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix{scale(matrix.data_, factor)};
        }

        friend constexpr Matrix operator*(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.columns() == right.rows());

            if (auto expected_product{product(left.data_, right.data_)}; expected_product.has_value()) {
                return Matrix{std::move(expected_product).value()};
            } else {
                print_error(expected_product.error());
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
            return Matrix{scale(matrix.data_, 1 / factor)};
        }

        friend constexpr Matrix operator/(Matrix const& left, Matrix const& right)
        {
            // assert correct dimensions
            assert(left.columns() == right.rows());

            // division is multiplication by inverse
            if (auto expected_inverse{inverse(right)}; expected_inverse.has_value()) {
                if (auto expected_product{product(left, expected_inverse.value())}; expected_product.has_value()) {
                    return Matrix{std::move(expected_product).value()};
                } else {
                    print_error(expected_product.error());
                    std::unreachable();
                }
            } else {
                print_error(expected_inverse.error());
                std::unreachable();
            }
        }

        friend constexpr Matrix operator^(Matrix const& matrix, Value const& factor)
        {
            assert(matrix.is_square());
            auto result{matrix.data_};
            for (Value i{}; i < factor - 1; ++i) {
                result *= matrix.data_;
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
            Matrix::print(self.data_);
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

        constexpr void swap(this Matrix& self, MatrixData& other)
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
            return self.data_[0].size();
        }

        constexpr VectorData diagonal(this Matrix const& self)
        {
            assert(self.rows() == self.columns());

            VectorData diagonale{};
            diagonale.reserve(self.rows());

            for (Size diag{0}; diag < self.rows(); ++diag) {
                diagonale.push_back(self.data_[diag][diag]);
            }
            return diagonale;
        }

        constexpr void transpose(this Matrix& self)
        {
            self.data_ = transposition(self.data_);
        }

        constexpr void invert(this Matrix& self)
        {
            if (auto expected_inverse{Matrix::inverse(self.data_)}; expected_inverse.has_value()) {
                self.data_ = std::move(expected_inverse).value();
            } else {
                print_error(expected_inverse.error());
                std::unreachable();
            }
        }

    private:
        static constexpr const char* matrix_error_to_string(MatrixError const MatrixError) noexcept
        {
            switch (MatrixError) {
                case MatrixError::WRONG_DIMS:
                    return "Wrong dims";
                case MatrixError::SINGULARITY:
                    return "Singularity";
                case MatrixError::BAD_ALLOC:
                    return "Bad alloc";
                default:
                    return "None";
            }
        }

        static constexpr void print_error(MatrixError const MatrixError) noexcept
        {
            fmt::print("%s", matrix_error_to_string(MatrixError));
        }

        static constexpr void print(MatrixData const& data) noexcept
        {
            fmt::print("[");

            auto row{data.cbegin()};
            while (row != data.cend()) {
                fmt::print("[");
                auto col{std::cbegin(*row)};
                while (col != std::cend(*row)) {
                    if constexpr (std::is_integral_v<Value>) {
                        fmt::print("%ld", static_cast<long int>(*col));
                    } else if constexpr (std::is_floating_point_v<Value>) {
                        fmt::print("%Lf", static_cast<long double>(*col));
                    }
                    if (col != std::cend(*row)) {
                        fmt::print(", ");
                    }
                    std::advance(col, 1);
                }
                fmt::print("]");
                if (std::next(row) != data.cend()) {
                    fmt::print(",\n");
                }
                std::advance(row, 1);
            }

            fmt::print("]\n");
        }

        static constexpr MatrixData make_matrix(std::initializer_list<const std::initializer_list<Value>> const data)
        {
            MatrixData matrix{};
            matrix.reserve(data.size());
            for (auto const& row : data) {
                auto& column{matrix.emplace_back()};
                column.reserve(row.size());
                for (auto const& col : row) {
                    column.push_back(col);
                }
            }
            return matrix;
        }

        static constexpr MatrixData make_zeros(Size const rows, Size const columns)
        {
            MatrixData matrix{};
            matrix.reserve(rows);
            for (Size row{0}; row < rows; ++row) {
                auto& column{matrix.emplace_back()};
                column.reserve(columns);
                for (Size col{0}; col < columns; ++col) {
                    column.emplace_back();
                }
            }
            return matrix;
        }

        static constexpr MatrixData make_ones(Size const rows, Size const columns)
        {
            MatrixData matrix{};
            matrix.reserve(rows);
            for (Size row{0}; row < rows; ++row) {
                auto& column{matrix.emplace_back()};
                column.reserve(columns);
                for (Size col{0}; col < columns; ++col) {
                    column.push_back(Value{1});
                }
            }
            return matrix;
        }

        static constexpr MatrixData make_diagonal(std::initializer_list<Value> const diagonal)
        {
            MatrixData matrix{};
            matrix.reserve(diagonal.size());
            for (Size row{0}; row < diagonal.size(); ++row) {
                auto& column{matrix.emplace_back()};
                column.reserve(diagonal.size());
                for (Size col{0}; col < diagonal.size(); ++col) {
                    if (col == row) {
                        column.push_back(*std::next(diagonal.begin(), col));
                    } else {
                        column.emplace_back();
                    }
                }
            }
            return matrix;
        }

        static constexpr MatrixData make_eye(Size const dimensions)
        {
            MatrixData matrix{};
            matrix.reserve(dimensions);
            for (Size row{0}; row < dimensions; ++row) {
                auto& column{matrix.emplace_back()};
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

        static constexpr MatrixData make_row(Size const rows)
        {
            VectorData row_vector{};
            row_vector.reserve(rows);
            for (Size row{}; row < rows; ++row) {
                row_vector.emplace_back();
            }
            return row_vector;
        }
        static constexpr MatrixData make_row(std::initializer_list<Value> const data)
        {
            VectorData row_vector{};
            auto const columns{data.size()};
            row_vector.reserve(columns);
            auto make_column{[column{data.begin()}]() -> decltype(auto) { return *(column)++; }};
            for (Size row{}; row < data.size(); ++row) {
                row_vector.push_back(make_column());
            }
            return row_vector;
        }

        static constexpr MatrixData make_column(Size const columns)
        {
            MatrixData column_vector{};
            auto& column{column_vector.emplace_back()};
            column.reserve(columns);
            for (Size col{}; col < columns; ++col) {
                column.emplace_back();
            }
            return column_vector;
        }

        static constexpr MatrixData make_column(std::initializer_list<Value> const data)
        {
            MatrixData column_vector{};
            auto& column{column_vector.emplace_back()};
            column.reserve(data.size());
            auto make_row{[row{data.begin()}]() -> decltype(auto) { return *(row)++; }};
            for (Size row{}; row < data.size(); ++row) {
                column_vector.push_back(make_row());
            }
            return column_vector;
        }

        static constexpr ExpectedMatrix
        minor(MatrixData const& data, Size const row, Size const column, Size const dimensions)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }
            // assert cofactor isnt calculated for minor bigger than data
            assert(dimensions <= row && dimensions <= column);
            if (dimensions > row || dimensions > column) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }
            // minor is scalar, can omit later code
            if (dimensions == 0) {
                return ExpectedMatrix{data};
            }

            auto minor{make_zeros(dimensions, dimensions)};
            Size cof_row{0};
            Size cof_column{0};
            for (Size row_{0}; row_ < dimensions; ++row_) {
                for (Size column_{0}; column_ < dimensions; ++column_) {
                    // copying into cofactor matrixonly those element which are not in given row and column
                    if (row_ != row && column_ != column) {
                        minor[cof_row][cof_column++] = data[row_][column_];

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

        static constexpr ExpectedDet determinant(MatrixData const& data, Size const dimensions)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // assert minor isnt bigger than data
            assert(rows >= dimensions && columns >= dimensions);
            if (rows < dimensions || columns < dimensions) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // data is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedDet{data[0][0]};
            }
            // data is 2x2 matrix, can omit later code
            if (dimensions == 2) {
                return ExpectedDet{std::in_place, (data[0][0] * data[1][1]) - (data[1][0] * data[0][1])};
            }

            auto det{static_cast<Value>(0)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size column{0}; column < dimensions; ++column) {
                // cofactor of data[0][column]

                if (auto expected_minor{Matrix::minor(data, 0, column, dimensions)}; expected_minor.has_value()) {
                    minor = std::move(expected_minor).value();
                } else {
                    print_error(expected_minor.error());
                    std::unreachable();
                }

                if (auto expected_det{determinant(minor, dimensions - 1)}; expected_det.has_value()) {
                    det += sign * data[0][column] * std::move(expected_det).value();
                } else {
                    print_error(expected_det.error());
                    std::unreachable();
                }

                // alternate sign
                sign *= static_cast<Value>(-1);
            }

            return ExpectedDet{det};
        }

        static constexpr MatrixData transposition(MatrixData const& data)
        {
            Size const new_rows{data.size()};
            Size const new_columns{data[0].size()};

            // data is scalar, can omit later code
            if ((new_rows == new_columns) == 1)
                return data;

            auto transposition{make_zeros(new_rows, new_columns)};
            for (Size row{0}; row < new_rows; ++row) {
                for (Size column{0}; column < new_columns; ++column) {
                    transposition[row][column] = data[column][row];
                }
            }
            return transposition;
        }

        static constexpr ExpectedMatrix adjoint(MatrixData const& data)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // data is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedMatrix{data};
            }

            auto complement{make_zeros(dimensions, dimensions)};

            // sign multiplier
            auto sign{static_cast<Value>(1)};
            auto minor{make_zeros(dimensions, dimensions)};
            for (Size row{0}; row < dimensions; ++row) {
                for (Size column{0}; column < dimensions; column++) {
                    // get cofactor of data[row][column]

                    if (auto expected_minor{Matrix::minor(data, row, column, dimensions)}; expected_minor.has_value()) {
                        minor = std::move(expected_minor).value();
                    } else {
                        print_error(expected_minor.error());
                        std::unreachable();
                    }

                    // sign of adj[column][row] positive if sum of row and column indexes is even
                    if ((row + column) % 2 == 0) {
                        sign = static_cast<Value>(1);
                    } else {
                        sign = static_cast<Value>(-1);
                    }

                    // complement is matrixof determinants of minors with alternating signs!!!
                    if (auto expected_det{determinant(minor, dimensions - 1)}; expected_det.has_value()) {
                        complement[row][column] = (sign)*std::move(expected_det).value();
                    } else {
                        print_error(expected_det.error());
                        std::unreachable();
                    }
                }
            }

            // adjoSize is transposed of complement matrix
            return ExpectedMatrix{transposition(complement)};
        }

        static constexpr ExpectedMatrix inverse(MatrixData const& data)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // data is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedMatrix{data};
            }

            if (auto expected_det{determinant(data, dimensions)}; expected_det.has_value()) {
                auto const det{std::move(expected_det).value()};

                // assert correct determinant
                assert(det != 0);
                if (det == 0) {
                    return Unexpected{MatrixError::SINGULARITY};
                }

                if (auto expected_adjoint{adjoint(data)}; expected_adjoint.has_value()) {
                    auto const adjoint{std::move(expected_adjoint).value()};

                    // inverse is adjoint matrixdivided by det factor
                    // division is multiplication by inverse
                    return ExpectedMatrix{scale(adjoint, 1 / det)};
                } else {
                    print_error(expected_adjoint.error());
                    std::unreachable();
                }
            } else {
                print_error(expected_det.error());
                std::unreachable();
            }
        }

        static constexpr ExpectedMatrix upper_triangular(MatrixData const& data)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions{rows};
            // data is scalar
            if (dimensions == 1)
                return ExpectedMatrix{data};

            // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
            return ExpectedMatrix{transposition(lower_triangular(data))};
        }

        static constexpr ExpectedMatrix lower_triangular(MatrixData const& data)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            Size const dimensions = rows; // = columns;
            // data is scalar
            if (dimensions == 1) {
                return ExpectedMatrix{data};
            }

            auto lower_triangular{make_zeros(dimensions, dimensions)};

            // decomposing data matrixinto lower triangular
            for (Size row{0}; row < dimensions; ++row) {
                for (Size column{0}; column <= row; ++column) {
                    Value sum{};

                    // summation for diagonals
                    if (column == row) {
                        for (Size sum_col{0}; sum_col < column; ++sum_col) {
                            sum += std::pow(lower_triangular[column][sum_col], 2);
                        }
                        lower_triangular[column][column] = std::sqrt(data[column][column] - sum);
                    } else {
                        // evaluating L(row, column) using L(column, column)
                        for (Size sum_col{0}; sum_col < column; ++sum_col) {
                            sum += (lower_triangular[row][sum_col] * lower_triangular[column][sum_col]);
                        }
                        lower_triangular[row][column] = (data[row][column] - sum) / lower_triangular[column][column];
                    }
                }
            }
            return ExpectedMatrix{std::move(lower_triangular)};
        }

        static constexpr ExpectedMatrix product(MatrixData const& left, MatrixData const& right)
        {
            Size const left_rows{left.size()};
            Size const right_rows{right.size()};
            Size const left_columns = {left[0].size()};
            Size const right_columns{right[0].size()};

            // assert correct dimensions
            assert(left_columns == right_rows);
            if (left_columns != right_rows) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            Size const product_rows{left_rows};
            Size const product_columns{right_columns};
            auto product{make_zeros(product_rows, product_columns)};

            for (Size left_row{0}; left_row < left_rows; ++left_row) {
                for (Size right_column{0}; right_column < right_columns; ++right_column) {
                    Value sum{0};
                    for (Size left_column{0}; left_column < left_columns; ++left_column) {
                        sum += left[left_row][left_column] * right[left_column][right_column];
                    }
                    product[left_row][right_column] = sum;
                }
            }
            return ExpectedMatrix{std::move(product)};
        }

        static constexpr MatrixData sum(MatrixData const& left, MatrixData const& right) noexcept
        {
            assert(left.size() == right.size());
            assert(left[0].size() == right[0].size());

            auto sum{make_zeros(left.size(), right.size())};
            for (Size row{0}; row < left.size(); ++row) {
                for (Size column{0}; column < left[0].size(); ++column) {
                    sum[row][column] = left[row][column] + right[row][column];
                }
            }
            return sum;
        }

        static constexpr MatrixData difference(MatrixData const& left, MatrixData const& right) noexcept
        {
            assert(left.size() == right.size());
            assert(left[0].size() == right[0].size());

            auto difference{make_zeros(left.size(), right.size())};
            for (Size row{0}; row < left.size(); ++row) {
                for (Size column{0}; column < left[0].size(); ++column) {
                    difference[row][column] = left[row][column] - right[row][column];
                }
            }
            return difference;
        }

        static constexpr MatrixData scale(MatrixData const& data, Value const factor)
        {
            Size const rows{data.size()};
            Size const columns{data[0].size()};

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return data;
            }
            // factor is 0 then return matrixof zeros
            else if (factor == 0) {
                return make_zeros(data.size(), data[0].size());
            }

            auto scale{make_zeros(rows, columns)};
            for (Size row{0}; row < rows; ++row) {
                for (Size column{0}; column < columns; ++column) {
                    scale[row][column] = data[row][column] * factor;
                }
            }
            return scale;
        }

        MatrixData data_{};
    };

}; // namespace Linalg

#endif // MATRIX_HPP