#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <cmath>
#include <compare>
#include <cstdio>
#include <expected>
#include <utility>
#include <vector>

namespace {
    enum class matrix_error {
        wrong_dims,
        singularity,
        bad_alloc,
    };

    constexpr auto matrix_error_to_string(const matrix_error error) noexcept
    {
        switch (error) {
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

    constexpr void log(const matrix_error error) noexcept
    {
        puts(matrix_error_to_string(error));
    }

    struct diagonal_t {
        explicit diagonal_t() = default;
    };
    constexpr diagonal_t diagonal_tag{};
}; // namespace

template <typename value_type>
class matrix_wrapper {
public:
    using vector = std::vector<value_type>;
    using matrix = std::vector<std::vector<value_type>>;

    [[nodiscard]] static constexpr matrix_wrapper<value_type> diagonal(const std::size_t matrix_dims,
                                                                       const value_type value)
    {
        return matrix_wrapper<value_type>{diagonal_tag, matrix_dims, value};
    }

    [[nodiscard]] static constexpr matrix_wrapper<value_type> eye(const std::size_t matrix_dims)
    {
        return matrix_wrapper<value_type>{diagonal_tag, matrix_dims, value_type{1}};
    }

    [[nodiscard]] static constexpr matrix_wrapper<value_type> ones(const std::size_t matrix_dims)
    {
        return matrix_wrapper<value_type>{matrix_dims, matrix_dims, value_type{1}};
    }

    [[nodiscard]] static constexpr matrix_wrapper<value_type> zeros(const std::size_t matrix_dims)
    {
        return matrix_wrapper<value_type>{matrix_dims, matrix_dims};
    }

    constexpr matrix_wrapper() noexcept = default;

    constexpr matrix_wrapper(const std::size_t rows, const std::size_t columns) :
        rows_{rows}, columns_{columns}, data_{make_matrix(rows, columns)}
    {
    }

    constexpr matrix_wrapper(const std::size_t rows, const std::size_t columns, const value_type value) :
        rows_{rows}, columns_{columns}, data_{make_matrix(rows, columns, value)}
    {
    }

    constexpr matrix_wrapper([[maybe_unused]] const diagonal_t diagonal_tag,
                             const std::size_t matrix_dims,
                             const value_type value) :
        rows_{matrix_dims}, columns_{matrix_dims}, data_{make_diagonal(matrix_dims, value)}
    {
    }

    explicit constexpr matrix_wrapper(matrix&& data) noexcept :
        rows_{data.size()}, columns_{data[0].size()}, data_{std::forward<matrix>(data)}
    {
    }

    explicit constexpr matrix_wrapper(const matrix& data) : rows_{data.size()}, columns_{data[0].size()}, data_{data}
    {
    }

    constexpr matrix_wrapper(const matrix_wrapper& other) = default;

    constexpr matrix_wrapper(matrix_wrapper&& other) noexcept = default;

    constexpr ~matrix_wrapper() noexcept = default;

    constexpr matrix_wrapper& operator=(const matrix_wrapper& other) = default;

    constexpr matrix_wrapper& operator=(matrix_wrapper&& other) noexcept = default;

    constexpr void operator=(matrix&& data) noexcept
    {
        this->data_ = std::forward<matrix>(data);
        this->rows_ = data.size();
        this->columns_ = data[0].size();
    }

    constexpr void operator=(const matrix& data)
    {
        this->data_ = data;
        this->rows_ = data.size();
        this->columns_ = data[0].size();
    }

    constexpr matrix_wrapper& operator+=(const matrix_wrapper& other) noexcept
    {
        // assert correct dimensions
        assert(other.rows_ == this->rows_);
        assert(other.columns_ == this->rows_);

        for (std::size_t row{0}; row < this->rows_; ++row) {
            for (std::size_t column{0}; column < this->columns_; ++column) {
                this->data_[row][column] += other.data_[row][column];
            }
        }
        return *this;
    }

    constexpr matrix_wrapper& operator-=(const matrix_wrapper& other) noexcept
    {
        // assert correct dimensions
        assert(other.rows_ == this->rows_);
        assert(other.columns_ == this->rows_);

        for (std::size_t row{0}; row < this->rows_; ++row) {
            for (std::size_t column{0}; column < this->columns_; ++column) {
                this->data_[row][column] -= other.data_[row][column];
            }
        }
        return *this;
    }

    constexpr matrix_wrapper& operator*=(const value_type& factor) noexcept
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return *this;
        }
        // factor is 0 then return zero-ed this
        else if (factor == 0) {
            this->clear();
            return *this;
        }

        this->data_ = scale(this->data_, factor);
        return *this;
    }

    constexpr matrix_wrapper& operator/=(const value_type& factor) noexcept
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

    constexpr matrix_wrapper& operator*=(const matrix_wrapper& other) noexcept
    {
        // assert correct dimensions
        assert(this->columns_ == other.rows_);

        this->data_ = product(this->data_, other.data_);
        // this->rows_ = this->rows_;
        this->columns_ = other.columns_;
        return *this;
    }

    constexpr matrix_wrapper& operator/=(const matrix_wrapper& other) noexcept
    {
        // assert correct dimensions
        assert(this->columns_ == other.rows_);

        // division is multiplication by inverse
        if (auto expected_inverse{inverse(other.data_)}; expected_inverse.has_value()) {
            if (auto expected_product{product(this->data_, expected_inverse.value())}; expected_product.has_value()) {
                *this = std::move(expected_product).value();
                this->columns_ = other.columns_;
                return *this;
            }
            else {
                log(expected_product.error());
                std::unreachable();
            }
        }
        else {
            log(expected_inverse.error());
            std::unreachable();
        }
    }

    constexpr matrix_wrapper operator+(const matrix_wrapper& other) const noexcept
    {
        // assert correct dimensions
        assert(other.rows_ == this->rows_);
        assert(other.columns_ == this->rows_);

        matrix_wrapper result{other.rows_, other.columns_};

        for (std::size_t row{0}; row < this->rows_; ++row) {
            for (std::size_t column{0}; column < this->columns_; ++column) {
                result.data_[row][column] = this->data_[row][column] + other.data_[row][column];
            }
        }
        return result;
    }

    constexpr matrix_wrapper operator-(const matrix_wrapper& other) const noexcept
    {
        // assert correct dimensions
        assert(other.rows_ == this->rows_);
        assert(other.columns_ == this->rows_);

        matrix_wrapper result{other.rows_, other.columns_};

        for (std::size_t row{0}; row < this->rows_; ++row) {
            for (std::size_t column{0}; column < this->columns_; ++column) {
                result.data_[row][column] = this->data_[row][column] - other.data_[row][column];
            }
        }
        return result;
    }

    friend constexpr matrix_wrapper operator*(const value_type& factor, const matrix_wrapper& matrix)
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return matrix;
        }
        // factor is 0 then return matrix of zeros
        else if (factor == 0) {
            return zeros(matrix.rows_, matrix.columns_);
        }
        return matrix_wrapper{scale(matrix.data_, factor)};
    }

    constexpr matrix_wrapper operator*(const value_type& factor) const
    {
        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return *this;
        }
        // factor is 0 then return matrix of zeros
        else if (factor == 0) {
            return zeros(this->rows_, this->columns_);
        }
        return matrix_wrapper{scale(this->data_, factor)};
    }

    constexpr matrix_wrapper operator/(const value_type& factor) const noexcept
    {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return *this;
        }

        // division is multiplication by inverse
        return matrix_wrapper{scale(this->data_, 1 / factor)};
    }

    constexpr matrix_wrapper operator*(const matrix_wrapper& other) const
    {
        // assert correct dimensions
        assert(this->columns_ == other.rows_);

        return matrix_wrapper{product(this->data_, other.data_)};
    }

    constexpr matrix_wrapper operator/(const matrix_wrapper& other) const
    {
        // assert correct dimensions
        assert(this->columns_ == other.rows_);

        // division is multiplication by inverse
        if (auto expected_inverse{inverse(other.data_)}; expected_inverse.has_value()) {
            if (auto expected_product{product(this->data_, expected_inverse.value())}; expected_product.has_value()) {
                return matrix_wrapper{std::move(expected_product).value()};
            }
            else {
                log(expected_product.error());
                std::unreachable();
            }
        }
        else {
            log(expected_inverse.error());
            std::unreachable();
        }
    }

    explicit constexpr operator matrix() && noexcept
    {
        return std::move(data_);
    }

    explicit constexpr operator matrix() const& noexcept
    {
        return data_;
    }

    [[nodiscard]] constexpr const vector& operator[](const std::size_t row) const noexcept
    {
        return this->data_[row];
    }

    [[nodiscard]] constexpr vector& operator[](const std::size_t row) noexcept
    {
        return this->data_[row];
    }

    [[nodiscard]] constexpr value_type& operator[](const std::size_t row, const std::size_t column) noexcept
    {
        return this->data_[row][column];
    }

    [[nodiscard]] constexpr const value_type& operator[](const std::size_t row, const std::size_t column) const noexcept
    {
        return this->data_[row][column];
    }

    [[nodiscard]] constexpr bool operator<=>(const matrix_wrapper& other) const noexcept = default;

    [[nodiscard]] constexpr const matrix& data() const& noexcept
    {
        return this->data_;
    }

    [[nodiscard]] constexpr matrix&& data() noexcept
    {
        std::move(this->data_);
    }

    constexpr void data(matrix&& data) noexcept
    {
        this->data_ = std::forward<matrix>(data);
        this->rows_ = data.size();
        this->columns_ = data[0].size();
    }

    constexpr void data(const matrix& data)
    {
        this->data_ = data;
        this->rows_ = data.size();
        this->columns_ = data[0].size();
    }

    constexpr void append_row(const std::size_t row, const vector& new_row)
    {
        assert(new_row.size() == this->columns_);
        this->data_.insert(std::next(this->data_.begin(), row), new_row); // basic pointer math (iterators are pointers)
        ++this->rows_;
    }

    constexpr void append_column(const std::size_t column, const vector& new_column)
    {
        assert(new_column.size() == this->rows_);
        for (std::size_t row{0}; row < this->rows_; ++row) {
            this->data_[row].insert(std::next(this->data_[row].begin(), column),
                                    new_column[row]); // basic pointer math (iterators are pointers)
        }
        ++this->columns_;
    }

    constexpr void delete_row(const std::size_t row)
    {
        assert(row <= this->rows_);
        this->data_.erase(std::next(this->data_.begin(), row));
        --this->rows_;
    }

    constexpr void delete_column(const std::size_t column)
    {
        assert(column <= this->columns_);
        for (std::size_t row{0}; row < this->rows_; ++row) {
            this->data_[row].erase(std::next(this->data_[row].begin(), column));
        }
        --this->columns_;
    }

    constexpr vector end_row() const noexcept
    {
        return this->data_.back();
    }

    constexpr vector begin_row() const noexcept
    {
        return this->data_.front();
    }

    constexpr vector end_column() const
    {
        vector end_column{};
        end_column.reserve(this->rows_);
        for (std::size_t row{0}; row < this->rows_; ++row) {
            end_column.push_back(data_[row].back());
        }
        return end_column;
    }

    constexpr vector begin_column() const
    {
        vector begin_column{};
        begin_column.reserve(this->rows_);
        for (std::size_t row{0}; row < this->rows_; ++row) {
            begin_column.push_back(data_[row].front());
        }
        return begin_column;
    }

    constexpr void reserve(const std::size_t new_rows, const std::size_t new_columns)
    {
        this->rows_ = new_rows;
        this->columns_ = new_columns;
        this->data_.reserve(this->rows_);
        for (auto& row : this->data_) {
            row.resever(this->columns_);
        }
    }

    constexpr void resize(const std::size_t new_rows, const std::size_t new_columns)
    {
        this->rows_ = new_rows;
        this->columns_ = new_columns;
        this->data_.resize(this->rows_);
        for (auto& row : this->data_) {
            row.resize(this->columns_);
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
        return this->rows_ == this->columns_ == 0;
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept
    {
        return this->rows_;
    }

    [[nodiscard]] constexpr std::size_t columns() const noexcept
    {
        return this->columns_;
    }

    constexpr vector diagonale() const
    {
        assert(this->rows_ == this->columns_);

        vector diagonale{};
        diagonale.reserve(this->rows_);

        for (std::size_t diag{0}; diag < this->rows_; ++diag) {
            diagonale.push_back(this->data_[diag][diag]);
        }
        return diagonale;
    }

    constexpr void transpose() noexcept
    {
        this->data_ = transposition(this->data_);

        // swap dimensions
        std::swap(this->rows_, this->columns_);
    }

    constexpr void invert() noexcept
    {
        if (auto expected_inverse{inverse(this->data_)}; expected_inverse.has_value()) {
            this->data_ = std::move(expected_inverse).value();
        }
        else {
            log(expected_inverse.error());
            std::unreachable();
        }
    }

private:
    static constexpr matrix make_matrix(const std::size_t rows, const std::size_t columns)
    {
        matrix matrix{};
        matrix.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(columns);
            for (std::size_t column_{0}; column_ < columns; ++column_) {
                column.emplace_back();
            }
        }
        return matrix;
    }

    static constexpr matrix make_matrix(const std::size_t rows, const std::size_t columns, const value_type value)
    {
        matrix matrix{};
        matrix.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(columns);
            for (std::size_t column_{0}; column_ < columns; ++column_) {
                column.push_back(value);
            }
        }
        return matrix;
    }

    static constexpr matrix make_diagonal(const std::size_t matrix_dims, const value_type value)
    {
        matrix matrix{};
        matrix.reserve(matrix_dims);
        for (std::size_t row{0}; row < matrix_dims; ++row) {
            auto& column{matrix.emplace_back()};
            column.reserve(matrix_dims);
            for (std::size_t col{0}; col < matrix_dims; ++col) {
                if (col == row) {
                    column.push_back(value);
                }
                else {
                    column.emplace_back();
                }
            }
        }
        return matrix;
    }

    static constexpr matrix make_vector(const std::size_t rows)
    {
        vector vector{};
        vector.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            vector.emplace_back();
        }
        return rows;
    }

    static constexpr matrix make_vector(const std::size_t rows, const value_type value)
    {
        vector vector{};
        vector.reserve(rows);
        for (std::size_t row{0}; row < rows; ++row) {
            vector.push_back(value);
        }
        return rows;
    }

    static constexpr std::expected<matrix, matrix_error>
    minor(const matrix& data, const std::size_t row, const std::size_t column, const std::size_t matrix_dims)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }
        // assert cofactor isnt calculated for minor bigger than data
        assert(matrix_dims < row && matrix_dims < column);
        if (matrix_dims >= row || matrix_dims >= column) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }
        // minor is scalar, can omit later code
        if (matrix_dims == 0) {
            return std::expected<matrix, matrix_error>{data};
        }

        auto minor{make_matrix(matrix_dims, matrix_dims)};
        std::size_t cof_row{0};
        std::size_t cof_column{0};
        for (std::size_t row_{0}; row_ < matrix_dims; ++row_) {
            for (std::size_t column_{0}; column_ < matrix_dims; ++column_) {
                // copying into cofactor matrix only those element which are not in given row and column
                if (row_ != row && column_ != column) {
                    minor[cof_row][cof_column++] = data[row_][column_];

                    // row is filled, so increase row index and reset column index
                    if (cof_column == matrix_dims - 1) {
                        cof_column = 0;
                        ++cof_row;
                    }
                }
            }
        }
        return std::expected<matrix, matrix_error>{std::move(minor)};
    }

    static constexpr std::expected<value_type, matrix_error> determinant(const matrix& data, std::size_t matrix_dims)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // assert minor isnt bigger than data
        assert(rows > matrix_dims && columns > matrix_dims);
        if (rows <= matrix_dims || columns <= matrix_dims) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // data is scalar, can omit later code
        if (matrix_dims == 1) {
            return std::expected<value_type, matrix_error>{data[0][0]};
        }
        // data is 2x2 matrix, can omit later code
        if (matrix_dims == 2) {
            return std::expected<value_type, matrix_error>{std::in_place,
                                                           (data[0][0] * data[1][1]) - (data[1][0] * data[0][1])};
        }

        auto det{static_cast<value_type>(0)};

        // sign multiplier
        auto sign{static_cast<value_type>(1)};
        auto minor{make_matrix(matrix_dims, matrix_dims)};
        for (std::size_t column{0}; column < matrix_dims; ++column) {
            // cofactor of data[0][column]

            if (auto expected_minor{minor(data, 0, column, matrix_dims)}; expected_minor.has_value()) {
                minor = std::move(expected_minor).value();
            }
            else {
                log(expected_minor.error());
                std::unreachable();
            }

            if (auto expected_det{determinant(minor, matrix_dims - 1)}; expected_det.has_value()) {
                det += sign * data[0][column] * std::move(expected_det).value();
            }
            else {
                log(expected_det.error());
                std::unreachable();
            }

            // alternate sign
            sign *= static_cast<value_type>(-1);
        }

        return std::expected<value_type, matrix_error>{det};
    }

    static constexpr matrix transposition(const matrix& data)
    {
        const std::size_t new_rows{data.size()};
        const std::size_t new_columns{data[0].size()};

        // data is scalar, can omit later code
        if ((new_rows == new_columns) == 1)
            return data;

        auto transposition{make_matrix(new_rows, new_columns)};
        for (std::size_t row{0}; row < new_rows; ++row) {
            for (std::size_t column{0}; column < new_columns; ++column) {
                transposition[row][column] = data[column][row];
            }
        }
        return transposition;
    }

    static constexpr std::expected<matrix, matrix_error> adjoint(const matrix& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrix square
        const std::size_t matrix_dims{rows};
        // data is scalar, can omit later code
        if (matrix_dims == 1) {
            return std::expected<matrix, matrix_error>{data};
        }

        auto complement{make_matrix(matrix_dims, matrix_dims)};

        // sign multiplier
        auto sign{static_cast<value_type>(1)};
        auto minor{make_matrix(matrix_dims, matrix_dims)};
        for (std::size_t row{0}; row < matrix_dims; ++row) {
            for (std::size_t column{0}; column < matrix_dims; column++) {
                // get cofactor of data[row][column]

                if (auto expected_minor{minor(data, row, column, matrix_dims)}; expected_minor.has_value()) {
                    minor = std::move(expected_minor).value();
                }
                else {
                    log(expected_minor.error());
                    std::unreachable();
                }

                // sign of adj[column][row] positive if sum of row and column indexes is even
                if ((row + column) % 2 == 0) {
                    sign = static_cast<value_type>(1);
                }
                else {
                    sign = static_cast<value_type>(-1);
                }

                // complement is matrix of determinants of minors with alternating signs!!!
                if (auto expected_det{determinant(minor, matrix_dims - 1)}; expected_det.has_value()) {
                    complement[row][column] = (sign)*std::move(expected_det).value();
                }
                else {
                    log(expected_det.error());
                    std::unreachable();
                }
            }
        }

        // adjostd::size_t is transposed of complement matrix
        return std::expected<matrix, matrix_error>{transposition(complement)};
    }

    static constexpr std::expected<matrix, matrix_error> inverse(const matrix& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrix square
        const std::size_t matrix_dims{rows};
        // data is scalar, can omit later code
        if (matrix_dims == 1) {
            return std::expected<matrix, matrix_error>{data};
        }

        if (auto expected_det{determinant(data, matrix_dims)}; expected_det.has_value()) {
            const auto det{std::move(expected_det).value()};

            // assert correct determinant
            assert(det != 0);
            if (det == 0) {
                return std::unexpected<matrix_error>{matrix_error::singularity};
            }

            if (auto expected_adjoint{adjoint(data)}; expected_adjoint.has_value()) {
                const auto adjoint{std::move(expected_adjoint).value()};

                // inverse is adjoint matrix divided by det factor
                // division is multiplication by inverse
                return std::expected<matrix, matrix_error>{scale(adjoint, 1 / det)};
            }
            else {
                log(expected_adjoint.error());
                std::unreachable();
            }
        }
        else {
            log(expected_det.error());
            std::unreachable();
        }
    }

    static constexpr std::expected<matrix, matrix_error> upper_triangular(const matrix& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrix square
        const std::size_t matrix_dims{rows};
        // data is scalar
        if (matrix_dims == 1)
            return std::expected<matrix, matrix_error>{data};

        // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
        return std::expected<matrix, matrix_error>{transposition(lower_triangular(data))};
    }

    static constexpr std::expected<matrix, matrix_error> lower_triangular(const matrix& data)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};
        // assert correct dimensions
        assert(rows == columns);
        if (rows != columns) {
            return std::unexpected<matrix_error>{matrix_error::wrong_dims};
        }

        // matrix square
        const std::size_t matrix_dims = rows; // = columns;
        // data is scalar
        if (matrix_dims == 1) {
            return std::expected<matrix, matrix_error>{data};
        }

        matrix lower_triangular{make_matrix(matrix_dims, matrix_dims)};

        // decomposing data matrix into lower triangular
        for (std::size_t row{0}; row < matrix_dims; ++row) {
            for (std::size_t column{0}; column <= row; ++column) {
                value_type sum{};

                // summation for diagonals
                if (column == row) {
                    for (std::size_t iSumCol{0}; iSumCol < column; ++iSumCol) {
                        sum += pow(lower_triangular[column][iSumCol], 2);
                    }
                    lower_triangular[column][column] = sqrt(data[column][column] - sum);
                }
                else {
                    // evaluating L(row, column) using L(column, column)
                    for (std::size_t iSumCol{0}; iSumCol < column; ++iSumCol) {
                        sum += (lower_triangular[row][iSumCol] * lower_triangular[column][iSumCol]);
                    }
                    lower_triangular[row][column] = (data[row][column] - sum) / lower_triangular[column][column];
                }
            }
        }
        return std::expected<matrix, matrix_error>{std::move(lower_triangular)};
    }

    static constexpr std::expected<matrix, matrix_error> product(const matrix& left, const matrix& right)
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
        auto product{make_matrix(product_rows, product_columns)};

        for (std::size_t left_row{0}; left_row < left_rows; ++left_row) {
            for (std::size_t right_column{0}; right_column < right_columns; ++right_column) {
                value_type sum{0};
                for (std::size_t left_column{0}; left_column < left_columns; ++left_column) {
                    sum += left[left_row][left_column] * right[left_column][right_column];
                }
                product[left_row][right_column] = sum;
            }
        }
        return std::expected<matrix, matrix_error>{std::move(product)};
    }

    static constexpr matrix scale(const matrix& data, const value_type factor)
    {
        const std::size_t rows{data.size()};
        const std::size_t columns{data[0].size()};

        // factor is 1 then dont need to do anything
        if (factor == 1) {
            return data;
        }
        // factor is 0 then return matrix of zeros
        else if (factor == 0) {
            data.clear();
            return data;
        }

        auto scale{make_matrix(rows, columns)};
        for (std::size_t row{0}; row < rows; ++row) {
            for (std::size_t column{0}; column < columns; ++column) {
                scale[row][column] = data[row][column] * factor;
            }
        }
        return scale;
    }

    std::size_t rows_{}; // first dim

    std::size_t columns_{}; // second dim

    matrix data_{}; // vector of vectors
};

#endif // MATRIX_HPP