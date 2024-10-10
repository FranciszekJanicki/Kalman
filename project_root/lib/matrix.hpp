#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdio>
#include <expected>
#include <initializer_list>
#include <utility>
#include <vector>

/* OVERVIEW:
    -create matrixes of given sizes using matrix::matrix(...) constructors (round bracket
    initialization) or using matrix::ones(...), matrix::zeros(...) and matrix::eye(...)  factory
    functions

    -create matrixes of given data using matrix::matrix{...} constructors (curly bracket initialization)
    or using matrix::matrix(...), matrix::row(...), matrix::column(...) factory functions
    (overloads with std::initializer_list, be careful, because {} init will always call these overloads)

    -create row, column vectors  and diagonal matrixes with given data using matrix::matrix(tag, ...)
    constructors (round bracket initialiation, first param being tag) or using matrix::row(...),
    matrix::column(...) and matrix::diagonal(...) factory functions

    -assign data using operator= assingment operators or using matrix::data(...) member functions
    -access data using matrix() conversion operators or using matrix::data() member functions

    -transpose using matrix::transpose() and invert using matrix::inver() member functions (invertion
    will fail if not possible)

    -multiply matrixwith matrix_data, multiply scalar with matrixand matrixwith scalar,
    divide matrixwith matrix(same as multiplying by inverse), add matrixes, substract matrixes, of course
    all if dimensions are correct for each of these operations

    -you can print matrixusing matrix::print() member function (using printf(), can change to std::print
    if compiler supports it)

    -full interface for data structure (std::vector<std::vector<type>> here)

    -full constexpr support (remember that dynamic memory allocated at compile time, stays at compile time- if you want
    to perform some matrixcalculations at compile time and then get result to run time, use
    std::array<std::array<type>> and copy from data (matrix::data() accessors) to array, using
    matrix::rows() and matrix::cols() to specify std::arrays dimensions)
*/

template <std::floating_point value_type>
class matrix {
public:
    using vector_data = std::vector<value_type>;

    using matrix_data = std::vector<std::vector<value_type>>;

    [[nodiscard]] static constexpr matrix row(const std::initializer_list<value_type> row)
    {
        return matrix{make_row(row)};
    }

    [[nodiscard]] static constexpr matrix column(const std::initializer_list<value_type> column)
    {
        return matrix{make_column(column)};
    }

    [[nodiscard]] static constexpr matrix row(const std::size_t rows)
    {
        return matrix{make_row(rows)};
    }

    [[nodiscard]] static constexpr matrix column(const std::size_t columns)
    {
        return matrix{make_column(columns)};
    }

    [[nodiscard]] static constexpr matrix diagonal(const std::initializer_list<value_type> diagonal)
    {
        return matrix{make_diagonal(diagonal)};
    }

    [[nodiscard]] static constexpr matrix eye(const std::size_t dimensions)
    {
        return matrix{make_eye(dimensions)};
    }

    [[nodiscard]] static constexpr matrix ones(const std::size_t rows, const std::size_t columns)
    {
        return matrix{make_ones(rows, columns)};
    }

    [[nodiscard]] static constexpr matrix zeros(const std::size_t rows, const std::size_t columns)
    {
        return matrix{make_zeros(rows, columns)};
    }

    constexpr matrix() noexcept = default;

    explicit constexpr matrix(const std::initializer_list<const std::initializer_list<value_type>> data) : data_{data}
    {
    }

    constexpr matrix(const std::size_t rows, const std::size_t columns) : data_{make_zeros(rows, columns)}
    {
    }

    explicit constexpr matrix(matrix_data&& data) noexcept : data_{std::forward<matrix_data>(data)}
    {
    }

    explicit constexpr matrix(const matrix_data& data) : data_{data}
    {
    }

    constexpr matrix(const matrix& other) = default;

    constexpr matrix(matrix&& other) noexcept = default;

    constexpr ~matrix() noexcept = default;

    constexpr matrix& operator=(const matrix& other) = default;

    constexpr matrix& operator=(matrix&& other) noexcept = default;

    constexpr void operator=(matrix_data&& data) noexcept
    {
        this->data_ = std::forward<matrix_data>(data);
    }

    constexpr void operator=(const matrix_data& data)
    {
        this->data_ = data;
    }

    constexpr matrix& operator+=(const matrix& other) noexcept
    {
        // assert correct dimensions
        assert(other.rows() == this->rows());
        assert(other.columns() == this->rows());

        for (std::size_t row{0}; row < this->rows(); ++row) {
            for (std::size_t column{0}; column < this->columns(); ++column) {
                this->data_[row][column] += other.data_[row][column];
            }
        }
        return *this;
    }

    constexpr matrix& operator-=(const matrix& other) noexcept
    {
        // assert correct dimensions
        assert(other.rows() == this->rows());
        assert(other.columns() == this->rows());

        for (std::size_t row{0}; row < this->rows(); ++row) {
            for (std::size_t column{0}; column < this->columns(); ++column) {
                this->data_[row][column] -= other.data_[row][column];
            }
        }
        return *this;
    }

    constexpr matrix& operator*=(const value_type& factor)
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return *this;
        }

        this->data_ = scale(this->data_, factor);
        return *this;
    }

    constexpr matrix& operator/=(const value_type& factor)
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

    constexpr matrix& operator*=(const matrix& other)
    {
        // assert correct dimensions
        assert(this->columns() == other.rows());

        if (auto expected_product{product(this->data_, other.data_)}; expected_product.has_value()) {
            this->data_ = std::move(expected_product).value();
            return *this;
        } else {
            print_matrix_error(expected_product.error());
            std::unreachable();
        }
    }

    constexpr matrix& operator/=(const matrix& other)
    {
        // assert correct dimensions
        assert(this->columns() == other.rows());

        // division is multiplication by inverse
        if (auto expected_inverse{inverse(other.data_)}; expected_inverse.has_value()) {
            if (auto expected_product{product(this->data_, expected_inverse.value())}; expected_product.has_value()) {
                *this = std::move(expected_product).value();
                return *this;
            } else {
                print_matrix_error(expected_product.error());
                std::unreachable();
            }
        } else {
            print_matrix_error(expected_inverse.error());
            std::unreachable();
        }
    }

    friend constexpr matrix operator+(const matrix& left, const matrix& right)
    {
        // assert correct dimensions
        assert(left.rows() == right.rows());
        assert(left.columns() == right.columns());

        return matrix{sum(left.data_, right.data_)};
    }

    friend constexpr matrix operator-(const matrix& left, const matrix& right)
    {
        // assert correct dimensions
        assert(left.rows() == right.rows());
        assert(left.columns() == right.columns());

        return matrix{difference(left.data_, right.data_)};
    }

    friend constexpr matrix operator*(const value_type& factor, const matrix& matrix_data)
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return matrix_data;
        }

        return matrix{scale(matrix_data.data_, factor)};
    }

    friend constexpr matrix operator*(const matrix& matrix_data, const value_type& factor)
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return matrix_data;
        }

        return matrix{scale(matrix_data.data_, factor)};
    }

    friend constexpr matrix operator/(const matrix& matrix_data, const value_type& factor)
    {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return matrix_data;
        }

        // division is multiplication by inverse
        return matrix{scale(matrix_data.data_, 1 / factor)};
    }

    friend constexpr matrix operator*(const matrix& left, const matrix& right)
    {
        // assert correct dimensions
        assert(left.columns() == right.rows());

        if (auto expected_product{product(left.data_, right.data_)}; expected_product.has_value()) {
            return matrix{std::move(expected_product).value()};
        } else {
            print_matrix_error(expected_product.error());
            std::unreachable();
        }
    }

    friend constexpr matrix operator/(const matrix& left, const matrix& right)
    {
        // assert correct dimensions
        assert(left.columns() == right.rows());

        // division is multiplication by inverse
        if (auto expected_inverse{inverse(right)}; expected_inverse.has_value()) {
            if (auto expected_product{product(left, expected_inverse.value())}; expected_product.has_value()) {
                return matrix{std::move(expected_product).value()};
            } else {
                print_matrix_error(expected_product.error());
                std::unreachable();
            }
        } else {
            print_matrix_error(expected_inverse.error());
            std::unreachable();
        }
    }

    explicit constexpr operator matrix() && noexcept
    {
        return std::move(this->data_);
    }

    explicit constexpr operator matrix() const& noexcept
    {
        return this->data_;
    }

    [[nodiscard]] constexpr const vector_data& operator[](const std::size_t row) const noexcept
    {
        assert(row <= this->rows());
        return this->data_[row];
    }

    [[nodiscard]] constexpr vector_data& operator[](const std::size_t row) noexcept
    {
        assert(row <= this->rows());
        return this->data_[row];
    }

    [[nodiscard]] constexpr value_type& operator[](const std::size_t row, const std::size_t column) noexcept
    {
        assert(row <= this->rows());
        assert(column <= this->columns());
        return this->data_[row][column];
    }

    [[nodiscard]] constexpr const value_type& operator[](const std::size_t row, const std::size_t column) const noexcept
    {
        assert(row <= this->rows());
        assert(column <= this->columns());
        return this->data_[row][column];
    }

    [[nodiscard]] constexpr bool operator<=>(const matrix& other) const noexcept = default;

    constexpr void print() const noexcept
    {
        matrix::print(this->data_);
    }

    [[nodiscard]] constexpr const matrix_data& data() const& noexcept
    {
        return this->data_;
    }

    [[nodiscard]] constexpr matrix_data&& data() && noexcept
    {
        return std::move(this->data_);
    }

    constexpr void data(matrix_data&& data) noexcept
    {
        this->data_ = std::forward<matrix_data>(data);
    }

    constexpr void data(const matrix_data& data)
    {
        this->data_ = data;
    }

    constexpr void swap(matrix& other)
    {
        std::swap(this->data_, other.data_);
    }

    constexpr void insert_row(const std::size_t row, const vector_data& new_row)
    {
        assert(new_row.size() == this->columns());
        this->data_.insert(std::next(this->data_.begin(), row), new_row);
    }

    constexpr void insert_column(const std::size_t column, const vector_data& new_column)
    {
        assert(new_column.size() == this->rows());
        for (const auto& row : this->data_) {
            row.insert(std::next(row.begin(), column), new_column[column]);
        }
    }

    constexpr void delete_row(const std::size_t row)
    {
        assert(row <= this->rows());
        this->data_.erase(std::next(this->data_.begin(), row));
    }

    constexpr void delete_column(const std::size_t column)
    {
        assert(column <= this->columns());
        for (std::size_t row{0}; row < this->rows(); ++row) {
            this->data_[row].erase(std::next(this->data_[row].begin(), column));
        }
    }

    constexpr const vector_data& end_row() const noexcept
    {
        return this->data_.back();
    }

    constexpr vector_data& end_row() noexcept
    {
        return this->data_.back();
    }

    constexpr const vector_data& begin_row() const noexcept
    {
        return this->data_.front();
    }

    constexpr vector_data& begin_row() noexcept
    {
        return this->data_.front();
    }

    constexpr vector_data end_column() const
    {
        vector_data end_column{};
        end_column.reserve(this->columns());
        for (const auto& row : this->data_) {
            end_column.push_back(row.back());
        }
        return end_column;
    }

    constexpr vector_data begin_column() const
    {
        vector_data begin_column{};
        begin_column.reserve(this->columns());
        for (const auto& row : this->data_) {
            begin_column.push_back(row.front());
        }
        return begin_column;
    }

    constexpr void reserve(const std::size_t rows, const std::size_t columns)
    {
        this->data_.reserve(rows);
        for (auto& row : this->data_) {
            row.reserve(columns);
        }
    }

    constexpr void resize(const std::size_t rows, const std::size_t columns)
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

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return this->rows() == this->columns() == 0;
    }

    [[nodiscard]] constexpr bool square() const noexcept
    {
        return this->rows() == this->columns();
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept
    {
        return this->data_.size();
    }

    [[nodiscard]] constexpr std::size_t columns() const noexcept
    {
        return this->data_[0].size();
    }

    constexpr vector_data diagonal() const
    {
        assert(this->rows() == this->columns());

        vector_data diagonale{};
        diagonale.reserve(this->rows());

        for (std::size_t diag{0}; diag < this->rows(); ++diag) {
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
        if (auto expected_inverse{inverse(this->data_)}; expected_inverse.has_value()) {
            this->data_ = std::move(expected_inverse).value();
        } else {
            print_matrix_error(expected_inverse.error());
            std::unreachable();
        }
    }

private:
    enum class matrix_error {
        wrong_dims,
        singularity,
        bad_alloc,
    };

    static constexpr auto matrix_error_to_string(const matrix_error matrix_error) noexcept
    {
        switch (matrix_error) {
            case matrix_error::wrong_dims:
                return "Wrong dims";
            case matrix_error::singularity:
                return "Singularity";
            case matrix_error::bad_alloc:
                return "Bad alloc";
            default:
                return "None";
        }
    }

    static constexpr void print_matrix_error(const matrix_error matrix_error) noexcept
    {
        printf("%s", matrix_error_to_string(matrix_error));
    }

    static constexpr void print(const matrix_data& data) noexcept
    {
        std::printf("[");

        auto row{data.cbegin()};
        while (row != data.cend()) {
            std::printf("[");
            auto col{std::cbegin(*row)};
            while (col != std::cend(*row)) {
                std::printf("%f", *col);
                if (col != std::cend(*row)) {
                    std::printf(", ");
                }
                std::advance(col, 1);
            }
            std::printf("]");
            if (std::next(row) != data.cend()) {
                std::printf(",\n");
            }
            std::advance(row, 1);
        }

        std::printf("]\n");
    }

    static constexpr matrix_data make_matrix(const std::initializer_list<const std::initializer_list<value_type>> data)
    {
        matrix_data matrix{};
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

    static constexpr matrix_data make_zeros(const std::size_t rows, const std::size_t columns)
    {
        matrix_data matrix{};
        matrix.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(columns);
            for (std::size_t col{0}; col < columns; ++col) {
                column.emplace_back();
            }
        }
        return matrix;
    }

    static constexpr matrix_data make_ones(const std::size_t rows, const std::size_t columns)
    {
        matrix_data matrix{};
        matrix.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(columns);
            for (std::size_t col{0}; col < columns; ++col) {
                column.push_back(value_type{1});
            }
        }
        return matrix;
    }

    static constexpr matrix_data make_diagonal(const std::initializer_list<value_type> diagonal)
    {
        matrix_data matrix{};
        matrix.reserve(diagonal.size());
        for (std::size_t row{0}; row < diagonal.size(); ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(diagonal.size());
            for (std::size_t col{0}; col < diagonal.size(); ++col) {
                if (col == row) {
                    column.push_back(*std::next(diagonal.begin(), col));
                } else {
                    column.emplace_back();
                }
            }
        }
        return matrix;
    }

    static constexpr matrix_data make_eye(const std::size_t dimensions)
    {
        matrix_data matrix{};
        matrix.reserve(dimensions);
        for (std::size_t row{0}; row < dimensions; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(dimensions);
            for (std::size_t col{0}; col < dimensions; ++col) {
                if (col == row) {
                    column.push_back(value_type{1});
                } else {
                    column.emplace_back();
                }
            }
        }
        return matrix;
    }

    static constexpr matrix_data make_row(const std::size_t rows)
    {
        vector_data row_vector{};
        row_vector.reserve(rows);
        for (std::size_t row{}; row < rows; ++row) {
            row_vector.emplace_back();
        }
        return row_vector;
    }

    static constexpr matrix_data make_row(const std::initializer_list<value_type> data)
    {
        vector_data row_vector{};
        const auto columns{data.size()};
        row_vector.reserve(columns);
        auto make_column{[column{data.begin()}]() -> decltype(auto) { return *(column)++; }};
        for (std::size_t row{}; row < data.size(); ++row) {
            row_vector.push_back(make_column());
        }
        return row_vector;
    }

    static constexpr matrix_data make_column(const std::size_t columns)
    {
        matrix_data column_vector{};
        auto& column{column_vector.emplace_back()};
        column.reserve(columns);
        for (std::size_t col{}; col < columns; ++col) {
            column.emplace_back();
        }
        return column_vector;
    }

    static constexpr matrix_data make_column(const std::initializer_list<value_type> data)
    {
        matrix_data column_vector{};
        auto& column{column_vector.emplace_back()};
        column.reserve(data.size());
        auto make_row{[row{data.begin()}]() -> decltype(auto) { return *(row)++; }};
        for (std::size_t row{}; row < data.size(); ++row) {
            column_vector.push_back(make_row());
        }
        return column_vector;
    }

    static constexpr std::expected<matrix_data, matrix_error>
    minor(const matrix_data& data, const std::size_t row, const std::size_t column, const std::size_t dimensions)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }
        // assert cofactor isnt calculated for minor bigger than data
        assert(dimensions <= row && dimensions <= column);
        if (dimensions > row || dimensions > column) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }
        // minor is scalar, can omit later code
        if (dimensions == 0) {
            return std::expected<matrix_data, matrix_error>{data};
        }

        auto minor{make_zeros(dimensions, dimensions)};
        std::size_t cof_row{0};
        std::size_t cof_column{0};
        for (std::size_t row_{0}; row_ < dimensions; ++row_) {
            for (std::size_t column_{0}; column_ < dimensions; ++column_) {
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
        return std::expected<matrix_data, matrix_error>{std::move(minor)};
    }

    static constexpr std::expected<value_type, matrix_error> determinant(const matrix_data& data,
                                                                         std::size_t dimensions)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // assert minor isnt bigger than data
        assert(rows >= dimensions && columns >= dimensions);
        if (rows < dimensions || columns < dimensions) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // data is scalar, can omit later code
        if (dimensions == 1) {
            return std::expected<value_type, matrix_error>{data[0][0]};
        }
        // data is 2x2 matrix_data, can omit later code
        if (dimensions == 2) {
            return std::expected<value_type, matrix_error>{std::in_place,
                                                           (data[0][0] * data[1][1]) - (data[1][0] * data[0][1])};
        }

        auto det{static_cast<value_type>(0)};

        // sign multiplier
        auto sign{static_cast<value_type>(1)};
        auto minor{make_zeros(dimensions, dimensions)};
        for (std::size_t column{0}; column < dimensions; ++column) {
            // cofactor of data[0][column]

            if (auto expected_minor{matrix::minor(data, 0, column, dimensions)}; expected_minor.has_value()) {
                minor = std::move(expected_minor).value();
            } else {
                print_matrix_error(expected_minor.error());
                std::unreachable();
            }

            if (auto expected_det{determinant(minor, dimensions - 1)}; expected_det.has_value()) {
                det += sign * data[0][column] * std::move(expected_det).value();
            } else {
                print_matrix_error(expected_det.error());
                std::unreachable();
            }

            // alternate sign
            sign *= static_cast<value_type>(-1);
        }

        return std::expected<value_type, matrix_error>{det};
    }

    static constexpr matrix_data transposition(const matrix_data& data)
    {
        const std::size_t new_rows{data.size()};
        const std::size_t new_columns{data[0].size()};

        // data is scalar, can omit later code
        if ((new_rows == new_columns) == 1)
            return data;

        auto transposition{make_zeros(new_rows, new_columns)};
        for (std::size_t row{0}; row < new_rows; ++row) {
            for (std::size_t column{0}; column < new_columns; ++column) {
                transposition[row][column] = data[column][row];
            }
        }
        return transposition;
    }

    static constexpr std::expected<matrix_data, matrix_error> adjoint(const matrix_data& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrixsquare
        const std::size_t dimensions{rows};
        // data is scalar, can omit later code
        if (dimensions == 1) {
            return std::expected<matrix_data, matrix_error>{data};
        }

        auto complement{make_zeros(dimensions, dimensions)};

        // sign multiplier
        auto sign{static_cast<value_type>(1)};
        auto minor{make_zeros(dimensions, dimensions)};
        for (std::size_t row{0}; row < dimensions; ++row) {
            for (std::size_t column{0}; column < dimensions; column++) {
                // get cofactor of data[row][column]

                if (auto expected_minor{matrix::minor(data, row, column, dimensions)}; expected_minor.has_value()) {
                    minor = std::move(expected_minor).value();
                } else {
                    print_matrix_error(expected_minor.error());
                    std::unreachable();
                }

                // sign of adj[column][row] positive if sum of row and column indexes is even
                if ((row + column) % 2 == 0) {
                    sign = static_cast<value_type>(1);
                } else {
                    sign = static_cast<value_type>(-1);
                }

                // complement is matrixof determinants of minors with alternating signs!!!
                if (auto expected_det{determinant(minor, dimensions - 1)}; expected_det.has_value()) {
                    complement[row][column] = (sign)*std::move(expected_det).value();
                } else {
                    print_matrix_error(expected_det.error());
                    std::unreachable();
                }
            }
        }

        // adjostd::size_t is transposed of complement matrix
        return std::expected<matrix_data, matrix_error>{transposition(complement)};
    }

    static constexpr std::expected<matrix_data, matrix_error> inverse(const matrix_data& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrixsquare
        const std::size_t dimensions{rows};
        // data is scalar, can omit later code
        if (dimensions == 1) {
            return std::expected<matrix_data, matrix_error>{data};
        }

        if (auto expected_det{determinant(data, dimensions)}; expected_det.has_value()) {
            const auto det{std::move(expected_det).value()};

            // assert correct determinant
            assert(det != 0);
            if (det == 0) {
                return std::unexpected<matrix_error>{matrix_error::singularity};
            }

            if (auto expected_adjoint{adjoint(data)}; expected_adjoint.has_value()) {
                const auto adjoint{std::move(expected_adjoint).value()};

                // inverse is adjoint matrixdivided by det factor
                // division is multiplication by inverse
                return std::expected<matrix_data, matrix_error>{scale(adjoint, 1 / det)};
            } else {
                print_matrix_error(expected_adjoint.error());
                std::unreachable();
            }
        } else {
            print_matrix_error(expected_det.error());
            std::unreachable();
        }
    }

    static constexpr std::expected<matrix_data, matrix_error> upper_triangular(const matrix_data& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrixsquare
        const std::size_t dimensions{rows};
        // data is scalar
        if (dimensions == 1)
            return std::expected<matrix_data, matrix_error>{data};

        // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
        return std::expected<matrix_data, matrix_error>{transposition(lower_triangular(data))};
    }

    static constexpr std::expected<matrix_data, matrix_error> lower_triangular(const matrix_data& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrixsquare
        const std::size_t dimensions = rows; // = columns;
        // data is scalar
        if (dimensions == 1) {
            return std::expected<matrix_data, matrix_error>{data};
        }

        auto lower_triangular{make_zeros(dimensions, dimensions)};

        // decomposing data matrixinto lower triangular
        for (std::size_t row{0}; row < dimensions; ++row) {
            for (std::size_t column{0}; column <= row; ++column) {
                value_type sum{};

                // summation for diagonals
                if (column == row) {
                    for (std::size_t sum_col{0}; sum_col < column; ++sum_col) {
                        sum += std::pow(lower_triangular[column][sum_col], 2);
                    }
                    lower_triangular[column][column] = std::sqrt(data[column][column] - sum);
                } else {
                    // evaluating L(row, column) using L(column, column)
                    for (std::size_t sum_col{0}; sum_col < column; ++sum_col) {
                        sum += (lower_triangular[row][sum_col] * lower_triangular[column][sum_col]);
                    }
                    lower_triangular[row][column] = (data[row][column] - sum) / lower_triangular[column][column];
                }
            }
        }
        return std::expected<matrix_data, matrix_error>{std::move(lower_triangular)};
    }

    static constexpr std::expected<matrix_data, matrix_error> product(const matrix_data& left, const matrix_data& right)
    {
        const std::size_t left_rows{left.size()};
        const std::size_t right_rows{right.size()};
        const std::size_t left_columns = {left[0].size()};
        const std::size_t right_columns{right[0].size()};

        // assert correct dimensions
        assert(left_columns == right_rows);
        if (left_columns != right_rows) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        const std::size_t product_rows{left_rows};
        const std::size_t product_columns{right_columns};
        auto product{make_zeros(product_rows, product_columns)};

        for (std::size_t left_row{0}; left_row < left_rows; ++left_row) {
            for (std::size_t right_column{0}; right_column < right_columns; ++right_column) {
                value_type sum{0};
                for (std::size_t left_column{0}; left_column < left_columns; ++left_column) {
                    sum += left[left_row][left_column] * right[left_column][right_column];
                }
                product[left_row][right_column] = sum;
            }
        }
        return std::expected<matrix_data, matrix_error>{std::move(product)};
    }

    static constexpr matrix_data sum(const matrix_data& left, const matrix_data& right) noexcept
    {
        assert(left.size() == right.size());
        assert(left[0].size() == right[0].size());

        auto sum{make_zeros(left.size(), right.size())};
        for (std::size_t row{0}; row < left.size(); ++row) {
            for (std::size_t column{0}; column < left[0].size(); ++column) {
                sum[row][column] = left[row][column] + right[row][column];
            }
        }
        return sum;
    }

    static constexpr matrix_data difference(const matrix_data& left, const matrix_data& right) noexcept
    {
        assert(left.size() == right.size());
        assert(left[0].size() == right[0].size());

        auto difference{make_zeros(left.size(), right.size())};
        for (std::size_t row{0}; row < left.size(); ++row) {
            for (std::size_t column{0}; column < left[0].size(); ++column) {
                difference[row][column] = left[row][column] - right[row][column];
            }
        }
        return difference;
    }

    static constexpr matrix_data scale(const matrix_data& data, const value_type factor)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};

        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return data;
        }
        // factor is 0 then return matrixof zeros
        else if (factor == 0) {
            return make_zeros(data.size(), data[0].size());
        }

        auto scale{make_zeros(rows, columns)};
        for (std::size_t row{0}; row < rows; ++row) {
            for (std::size_t column{0}; column < columns; ++column) {
                scale[row][column] = data[row][column] * factor;
            }
        }
        return scale;
    }

    matrix_data data_{}; // vector_data of vectors
};

#endif // MATRIX_HPP