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

        [[nodiscard]] static constexpr Matrix row(const std::initializer_list<Value> row)
        {
            return Matrix{make_row(row)};
        }

        [[nodiscard]] static constexpr Matrix row(const Size rows)
        {
            return Matrix{make_row(rows)};
        }

        [[nodiscard]] static constexpr Matrix column(const std::initializer_list<Value> column)
        {
            return Matrix{make_column(column)};
        }

        [[nodiscard]] static constexpr Matrix column(const Size columns)
        {
            return Matrix{make_column(columns)};
        }

        [[nodiscard]] static constexpr Matrix diagonal(const std::initializer_list<Value> diagonal)
        {
            return Matrix{make_diagonal(diagonal)};
        }

        [[nodiscard]] static constexpr Matrix eye(const Size dimensions)
        {
            return Matrix{make_eye(dimensions)};
        }

        [[nodiscard]] static constexpr Matrix ones(const Size rows, const Size columns)
        {
            return Matrix{make_ones(rows, columns)};
        }

        [[nodiscard]] static constexpr Matrix zeros(const Size rows, const Size columns)
        {
            return Matrix{make_zeros(rows, columns)};
        }

        constexpr Matrix() noexcept = default;

        explicit constexpr Matrix(const std::initializer_list<const std::initializer_list<Value>> data) : data_{data}
        {
        }

        constexpr Matrix(const Size rows, const Size columns) : data_{make_zeros(rows, columns)}
        {
        }

        explicit constexpr Matrix(MatrixData&& data) noexcept : data_{std::forward<MatrixData>(data)}
        {
        }

        explicit constexpr Matrix(const MatrixData& data) : data_{data}
        {
        }

        constexpr Matrix(const Matrix& other) = default;

        constexpr Matrix(Matrix&& other) noexcept = default;

        constexpr ~Matrix() noexcept = default;

        constexpr Matrix& operator=(const Matrix& other) = default;

        constexpr Matrix& operator=(Matrix&& other) noexcept = default;

        constexpr void operator=(MatrixData&& data) noexcept
        {
            this->data_ = std::forward<MatrixData>(data);
        }

        constexpr void operator=(const MatrixData& data)
        {
            this->data_ = data;
        }

        constexpr Matrix& operator+=(const Matrix other) noexcept
        {
            // assert correct dimensions
            assert(other.rows() == this->rows());
            assert(other.columns() == this->rows());

            for (Size row{0}; row < this->rows(); ++row) {
                for (Size column{0}; column < this->columns(); ++column) {
                    this->data_[row][column] += other.data_[row][column];
                }
            }
            return *this;
        }

        constexpr Matrix& operator-=(const Matrix& other) noexcept
        {
            // assert correct dimensions
            assert(other.rows() == this->rows());
            assert(other.columns() == this->rows());

            for (Size row{0}; row < this->rows(); ++row) {
                for (Size column{0}; column < this->columns(); ++column) {
                    this->data_[row][column] -= other.data_[row][column];
                }
            }
            return *this;
        }

        constexpr Matrix& operator*=(const Value& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return *this;
            }

            this->data_ = scale(this->data_, factor);
            return *this;
        }

        constexpr Matrix& operator*=(const Matrix& other)
        {
            // assert correct dimensions
            assert(this->columns() == other.rows());

            if (auto expected_product{Matrix::product(this->data_, other.data_)}; expected_product.has_value()) {
                this->data_ = std::move(expected_product).value();
                return *this;
            } else {
                print_error(expected_product.error());
                std::unreachable();
            }
        }

        constexpr Matrix& operator/=(const Value& factor)
        {
            // assert no division by 0!!!
            assert(factor != 0);

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return *this;
            }

            // division is multiplication by inverse
            this->data_ = scale(this->data_, 1 / factor);
            return *this;
        }

        constexpr Matrix& operator/=(const Matrix& other)
        {
            // assert correct dimensions
            assert(this->columns() == other.rows());

            // division is multiplication by inverse
            if (auto expected_inverse{inverse(other.data_)}; expected_inverse.has_value()) {
                if (auto expected_product{product(this->data_, expected_inverse.value())};
                    expected_product.has_value()) {
                    *this = std::move(expected_product).value();
                    return *this;
                } else {
                    print_error(expected_product.error());
                    std::unreachable();
                }
            } else {
                print_error(expected_inverse.error());
                std::unreachable();
            }
        }

        constexpr Matrix& operator^=(const Value& factor)
        {
            assert(this->is_square());
            for (Value i{}; i < factor - 1; ++i) {
                this->data_ *= this->data_;
            }
            return *this;
        }

        friend constexpr Matrix operator+(const Matrix& left, const Matrix& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            return Matrix{sum(left.data_, right.data_)};
        }

        friend constexpr Matrix operator-(const Matrix& left, const Matrix& right)
        {
            // assert correct dimensions
            assert(left.rows() == right.rows());
            assert(left.columns() == right.columns());

            return Matrix{difference(left.data_, right.data_)};
        }

        friend constexpr Matrix operator*(const Value& factor, const Matrix& matrix)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix{scale(matrix.data_, factor)};
        }

        friend constexpr Matrix operator*(const Matrix& matrix, const Value& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return matrix;
            }

            return Matrix{scale(matrix.data_, factor)};
        }

        friend constexpr Matrix operator*(const Matrix& left, const Matrix& right)
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

        friend constexpr Matrix operator/(const Matrix& matrix, const Value& factor)
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

        friend constexpr Matrix operator/(const Matrix& left, const Matrix& right)
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

        friend constexpr Matrix operator^(const Matrix& matrix, const Value& factor)
        {
            assert(matrix.is_square());
            auto result{matrix.data_};
            for (Value i{}; i < factor - 1; ++i) {
                result *= matrix.data_;
            }
            return result;
        }

        explicit constexpr operator MatrixData() && noexcept
        {
            return std::forward<Matrix>(*this).data_;
        }

        explicit constexpr operator MatrixData() const& noexcept
        {
            return this->data_;
        }

        [[nodiscard]] constexpr const VectorData& operator[](const Size row) const noexcept
        {
            assert(row <= this->rows());
            return this->data_[row];
        }

        [[nodiscard]] constexpr VectorData& operator[](const Size row) noexcept
        {
            assert(row <= this->rows());
            return this->data_[row];
        }

        [[nodiscard]] constexpr Value& operator[](const Size row, const Size column) noexcept
        {
            assert(row <= this->rows());
            assert(column <= this->columns());
            return this->data_[row][column];
        }

        [[nodiscard]] constexpr const Value& operator[](const Size row, const Size column) const noexcept
        {
            assert(row <= this->rows());
            assert(column <= this->columns());
            return this->data_[row][column];
        }

        [[nodiscard]] constexpr bool operator<=>(const Matrix& other) const noexcept = default;

        constexpr void print() const noexcept
        {
            Matrix::print(this->data_);
        }

        [[nodiscard]] constexpr const MatrixData& data() const& noexcept
        {
            return this->data_;
        }

        [[nodiscard]] constexpr MatrixData&& data() && noexcept
        {
            return std::forward<Matrix>(*this).data_;
        }

        constexpr void data(MatrixData&& data) noexcept
        {
            this->data_ = std::forward<MatrixData>(data);
        }

        constexpr void data(const MatrixData& data)
        {
            this->data_ = data;
        }

        constexpr void swap(MatrixData& other)
        {
            std::swap(this->data_, other.data_);
        }

        constexpr void insert_row(const Size row, const VectorData& new_row)
        {
            assert(new_row.size() == this->columns());
            this->data_.insert(std::next(this->data_.begin(), row), new_row);
        }

        constexpr void insert_column(const Size column, const VectorData& new_column)
        {
            assert(new_column.size() == this->rows());
            for (const auto& row : this->data_) {
                row.insert(std::next(row.begin(), column), new_column[column]);
            }
        }

        constexpr void delete_row(const Size row)
        {
            assert(row <= this->rows());
            this->data_.erase(std::next(this->data_.begin(), row));
        }

        constexpr void delete_column(const Size column)
        {
            assert(column <= this->columns());
            for (Size row{0}; row < this->rows(); ++row) {
                this->data_[row].erase(std::next(this->data_[row].begin(), column));
            }
        }

        constexpr const VectorData& end_row() const noexcept
        {
            return this->data_.back();
        }

        constexpr VectorData& end_row() noexcept
        {
            return this->data_.back();
        }

        constexpr const VectorData& begin_row() const noexcept
        {
            return this->data_.front();
        }

        constexpr VectorData& begin_row() noexcept
        {
            return this->data_.front();
        }

        constexpr VectorData end_column() const
        {
            VectorData end_column{};
            end_column.reserve(this->columns());
            for (const auto& row : this->data_) {
                end_column.push_back(row.back());
            }
            return end_column;
        }

        constexpr VectorData begin_column() const
        {
            VectorData begin_column{};
            begin_column.reserve(this->columns());
            for (const auto& row : this->data_) {
                begin_column.push_back(row.front());
            }
            return begin_column;
        }

        constexpr void reserve(const Size rows, const Size columns)
        {
            this->data_.reserve(rows);
            for (auto& row : this->data_) {
                row.reserve(columns);
            }
        }

        constexpr void resize(const Size rows, const Size columns)
        {
            this->data_.resize(rows);
            for (auto& row : this->data_) {
                row.resize(columns);
            }
        }

        constexpr void erase()
        {
            this->data_.erase();
        }

        constexpr void clear()
        {
            this->data_.clear();
        }

        [[nodiscard]] constexpr bool is_empty() const noexcept
        {
            return this->rows() == this->columns() == 0;
        }

        [[nodiscard]] constexpr bool is_square() const noexcept
        {
            return this->rows() == this->columns();
        }

        [[nodiscard]] constexpr Size rows() const noexcept
        {
            return this->data_.size();
        }

        [[nodiscard]] constexpr Size columns() const noexcept
        {
            return this->data_[0].size();
        }

        constexpr VectorData diagonal() const
        {
            assert(this->rows() == this->columns());

            VectorData diagonale{};
            diagonale.reserve(this->rows());

            for (Size diag{0}; diag < this->rows(); ++diag) {
                diagonale.push_back(this->data_[diag][diag]);
            }
            return diagonale;
        }

        constexpr void transpose()
        {
            this->data_ = transposition(this->data_);
        }

        constexpr void invert()
        {
            if (auto expected_inverse{Matrix::inverse(this->data_)}; expected_inverse.has_value()) {
                this->data_ = std::move(expected_inverse).value();
            } else {
                print_error(expected_inverse.error());
                std::unreachable();
            }
        }

    private:
        static constexpr const char* matrix_error_to_string(const MatrixError MatrixError) noexcept
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

        static constexpr void print_error(const MatrixError MatrixError) noexcept
        {
            fmt::print("%s", matrix_error_to_string(MatrixError));
        }

        static constexpr void print(const MatrixData& data) noexcept
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

        static constexpr MatrixData make_matrix(const std::initializer_list<const std::initializer_list<Value>> data)
        {
            MatrixData matrix{};
            matrix.reserve(data.size());
            for (const auto& row : data) {
                auto& column{matrix.emplace_back()};
                column.reserve(row.size());
                for (const auto& col : row) {
                    column.push_back(col);
                }
            }
            return matrix;
        }

        static constexpr MatrixData make_zeros(const Size rows, const Size columns)
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

        static constexpr MatrixData make_ones(const Size rows, const Size columns)
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

        static constexpr MatrixData make_diagonal(const std::initializer_list<Value> diagonal)
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

        static constexpr MatrixData make_eye(const Size dimensions)
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

        static constexpr MatrixData make_row(const Size rows)
        {
            VectorData row_vector{};
            row_vector.reserve(rows);
            for (Size row{}; row < rows; ++row) {
                row_vector.emplace_back();
            }
            return row_vector;
        }
        static constexpr MatrixData make_row(const std::initializer_list<Value> data)
        {
            VectorData row_vector{};
            const auto columns{data.size()};
            row_vector.reserve(columns);
            auto make_column{[column{data.begin()}]() -> decltype(auto) { return *(column)++; }};
            for (Size row{}; row < data.size(); ++row) {
                row_vector.push_back(make_column());
            }
            return row_vector;
        }

        static constexpr MatrixData make_column(const Size columns)
        {
            MatrixData column_vector{};
            auto& column{column_vector.emplace_back()};
            column.reserve(columns);
            for (Size col{}; col < columns; ++col) {
                column.emplace_back();
            }
            return column_vector;
        }

        static constexpr MatrixData make_column(const std::initializer_list<Value> data)
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
        minor(const MatrixData& data, const Size row, const Size column, const Size dimensions)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
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

        static constexpr ExpectedDet determinant(const MatrixData& data, Size dimensions)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
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

        static constexpr MatrixData transposition(const MatrixData& data)
        {
            const Size new_rows{data.size()};
            const Size new_columns{data[0].size()};

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

        static constexpr ExpectedMatrix adjoint(const MatrixData& data)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            const Size dimensions{rows};
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

        static constexpr ExpectedMatrix inverse(const MatrixData& data)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            const Size dimensions{rows};
            // data is scalar, can omit later code
            if (dimensions == 1) {
                return ExpectedMatrix{data};
            }

            if (auto expected_det{determinant(data, dimensions)}; expected_det.has_value()) {
                const auto det{std::move(expected_det).value()};

                // assert correct determinant
                assert(det != 0);
                if (det == 0) {
                    return Unexpected{MatrixError::SINGULARITY};
                }

                if (auto expected_adjoint{adjoint(data)}; expected_adjoint.has_value()) {
                    const auto adjoint{std::move(expected_adjoint).value()};

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

        static constexpr ExpectedMatrix upper_triangular(const MatrixData& data)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            const Size dimensions{rows};
            // data is scalar
            if (dimensions == 1)
                return ExpectedMatrix{data};

            // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
            return ExpectedMatrix{transposition(lower_triangular(data))};
        }

        static constexpr ExpectedMatrix lower_triangular(const MatrixData& data)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};
            // assert correct dimensions
            assert(rows == columns);
            if (rows != columns) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            // matrixsquare
            const Size dimensions = rows; // = columns;
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

        static constexpr ExpectedMatrix product(const MatrixData& left, const MatrixData& right)
        {
            const Size left_rows{left.size()};
            const Size right_rows{right.size()};
            const Size left_columns = {left[0].size()};
            const Size right_columns{right[0].size()};

            // assert correct dimensions
            assert(left_columns == right_rows);
            if (left_columns != right_rows) {
                return Unexpected{MatrixError::WRONG_DIMS};
            }

            const Size product_rows{left_rows};
            const Size product_columns{right_columns};
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

        static constexpr MatrixData sum(const MatrixData& left, const MatrixData& right) noexcept
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

        static constexpr MatrixData difference(const MatrixData& left, const MatrixData& right) noexcept
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

        static constexpr MatrixData scale(const MatrixData& data, const Value factor)
        {
            const Size rows{data.size()};
            const Size columns{data[0].size()};

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