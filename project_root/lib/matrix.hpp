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

    static constexpr auto matrix_error_to_string(const matrix_error error) noexcept
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

    static constexpr void LOG(const matrix_error error) noexcept
    {
        puts(matrix_error_to_string(error));
    }

}; // namespace

template <typename value_type>
class matrix_wrapper {
private:
    typedef typename std::vector<value_type> vector;

    typedef typename std::vector<std::vector<value_type>> matrix;

public:
    constexpr matrix_wrapper() noexcept = default;

    explicit constexpr matrix_wrapper(const std::size_t rows, const std::size_t cols); // 0

    explicit constexpr matrix_wrapper(const matrix& data); // 1

    explicit constexpr matrix_wrapper(matrix&& data) noexcept; // 2

    explicit constexpr matrix_wrapper(const vector& diag); // 3

    constexpr void transpose();

    constexpr void invert();

    constexpr void append_row(const std::size_t row, const vector& new_row);

    constexpr void append_column(const std::size_t column, const vector& new_column);

    constexpr void delete_row(const std::size_t row);

    constexpr void delete_column(const std::size_t column);

    constexpr void resize(const std::size_t new_rows, const std::size_t new_cols);

    constexpr void clear() noexcept;

    [[nodiscard]] constexpr bool empty() const noexcept;

    [[nodiscard]] constexpr std::size_t rows() const noexcept;

    [[nodiscard]] constexpr std::size_t columns() const noexcept;

    [[nodiscard]] constexpr vector end_row() const;

    [[nodiscard]] constexpr vector begin_row() const;

    [[nodiscard]] constexpr vector end_column() const;

    [[nodiscard]] constexpr vector begin_column() const;

    [[nodiscard]] constexpr vector diagonale() const;

    [[nodiscard]] constexpr const matrix& data() const& noexcept;

    [[nodiscard]] constexpr matrix&& data() && noexcept;

    constexpr matrix_wrapper operator*(const value_type& factor) const;

    constexpr matrix_wrapper operator/(const value_type& factor) const;

    constexpr matrix_wrapper operator+(const matrix_wrapper& other) const;

    constexpr matrix_wrapper operator-(const matrix_wrapper& other) const;

    constexpr matrix_wrapper operator*(const matrix_wrapper& other) const;

    constexpr matrix_wrapper operator/(const matrix_wrapper& other) const;

    constexpr matrix_wrapper& operator*=(const value_type& factor);

    constexpr matrix_wrapper& operator/=(const value_type& factor);

    constexpr matrix_wrapper& operator+=(const matrix_wrapper& other);

    constexpr matrix_wrapper& operator-=(const matrix_wrapper& other);

    constexpr matrix_wrapper& operator*=(const matrix_wrapper& other);

    constexpr matrix_wrapper& operator/=(const matrix_wrapper& other);

    constexpr void operator=(matrix&& data) noexcept;

    constexpr void operator=(const matrix& data) noexcept;

    [[nodiscard]] constexpr vector

    operator()(const std::size_t row) const noexcept;

    [[nodiscard]] constexpr value_type operator()(const std::size_t row, const std::size_t column) const noexcept;

    [[nodiscard]] constexpr bool operator<=>(const matrix_wrapper& other) const noexcept = default;

private:
    static constexpr matrix make_matrix(const std::size_t rows, const std::size_t cols);

    constexpr matrix scale(const matrix& data, const value_type factor) const;

    constexpr matrix transposition(const matrix& data) const;

    constexpr std::expected<matrix, matrix_error> adjoint(const matrix& data) const;

    constexpr std::expected<matrix, matrix_error> inverse(const matrix& data) const;

    constexpr std::expected<matrix, matrix_error> lower_triangular(const matrix& data) const;

    constexpr std::expected<matrix, matrix_error> upper_triangular(const matrix& data) const;

    constexpr std::expected<matrix, matrix_error> product(const matrix& left, const matrix& right) const;

    constexpr std::expected<value_type, matrix_error> determinant(const matrix& data,
                                                                  const std::size_t matrix_dims) const;

    constexpr std::expected<matrix, matrix_error>
    minor(const matrix& data, const std::size_t row, const std::size_t column, const std::size_t matrix_dims) const;

    std::size_t rows_{}; // first dim

    std::size_t columns_{}; // second dim

    value_type det_{};

    matrix data_{}; // vector of vectors
};

template <typename value_type>
inline constexpr matrix matrix_wrapper<value_type>::make_matrix(const std::size_t rows, const std::size_t cols)
{
    matrix rows_cols{};
    rows_cols.reserve(rows);
    for (std::size_t row{0}; row < rows; ++row) {
        auto& column{rows_cols.emplace_back()};
        column.reserve(cols);
        for (std::size_t col{0}; col < cols; ++col) {
            column.emplace_back();
        }
    }
    return rows_cols;
}

template <typename value_type>
constexpr matrix_wrapper<value_type>::matrix_wrapper(const std::size_t rows, const std::size_t cols) :
    rows_{rows}, columns_{cols}, data_{make_matrix(rows, cols)}
{
}

template <typename value_type>
constexpr matrix_wrapper<value_type>::matrix_wrapper(const vector& diag) :
    rows_{diag.size()}, columns_{rows_}, data_{make_matrix(rows_, columns_)}
{
    // create diagonal matrix with zeros beside diagonale
    for (std::size_t row{0}; row < rows_; ++row) {
        for (std::size_t col{0}; col < columns_; ++col) {
            if (row == col) {
                data_[row][col] = diag[row];
            }
        }
    }
}

template <typename value_type>
constexpr matrix_wrapper<value_type>::matrix_wrapper(matrix&& data) noexcept :
    rows_{data.size()}, columns_{data[0].size()}, data_{std::forward<matrix>(data)}
{
}

template <typename value_type>
constexpr matrix_wrapper<value_type>::matrix_wrapper(const matrix& data) :
    rows_{data.size()}, columns_{data[0].size()}, data_{data}
{
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::minor(const matrix& data,
                                                                                const std::size_t row,
                                                                                const std::size_t column,
                                                                                const std::size_t matrix_dims) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
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

    matrix minor{make_matrix(matrix_dims, matrix_dims)};
    std::size_t cof_row{0};
    std::size_t cof_col{0};
    for (std::size_t data_row{0}; data_row < matrix_dims; ++data_row) {
        for (std::size_t data_col{0}; data_col < matrix_dims; ++data_col) {
            // copying into cofactor matrix only those element which are not in given row and column
            if (data_row != row && data_col != column) {
                minor[cof_row][cof_col++] = data[data_row][data_col];

                // row is filled, so increase row index and reset col index
                if (cof_col == matrix_dims - 1) {
                    cof_col = 0;
                    ++cof_row;
                }
            }
        }
    }
    return std::expected<matrix, matrix_error>{std::move(minor)};
}

template <typename value_type>
constexpr std::expected<value_type, matrix_error> matrix_wrapper<value_type>::determinant(const matrix& data,
                                                                                          std::size_t matrix_dims) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    // assert minor isnt bigger than data
    assert(rows > matrix_dims && cols > matrix_dims);
    if (rows <= matrix_dims || cols <= matrix_dims) {
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

    value_type det{};

    // sign multiplier
    value_type sign{static_cast<value_type>(1)};
    matrix minor(matrix_dims, vector(cols, matrix_dims));
    for (std::size_t col{0}; col < matrix_dims; ++col) {
        // cofactor of data[0][col]

        if (auto expected_minor{minor(data, 0, col, matrix_dims)}; expected_minor.has_value()) {
            minor = std::move(expected_minor).value();
        }
        else {
            LOG(expected_minor.error());
            std::unreachable();
        }

        if (auto expected_det{determinant(minor, matrix_dims - 1)}; expected_det.has_value()) {
            det += sign * data[0][col] * std::move(expected_det).value();
        }
        else {
            LOG(expected_det.error());
            std::unreachable();
        }

        // alternate sign
        sign *= static_cast<value_type>(-1);
    }

    return std::expected<value_type, matrix_error>{det};
}

template <typename value_type>
constexpr matrix matrix_wrapper<value_type>::transposition(const matrix& data) const
{
    const std::size_t new_rows{data.size()};
    const std::size_t new_cols{data[0].size()};

    // data is scalar, can omit later code
    if ((new_rows == new_cols) == 1)
        return data;

    matrix transposition{make_matrix(new_rows, new_cols)};
    for (std::size_t row{0}; row < new_rows; ++row) {
        for (std::size_t col{0}; col < new_cols; ++col) {
            transposition[row][col] = data[col][row];
        }
    }
    return transposition;
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::adjoint(const matrix& data) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    // matrix square
    const std::size_t matrix_dims{rows};
    // data is scalar, can omit later code
    if (matrix_dims == 1) {
        return std::expected<matrix, matrix_error>{data};
    }

    matrix complement(matrix_dims, vector(matrix_dims, 0));

    // sign multiplier
    value_type sign{static_cast<value_type>(1)};
    matrix minor{make_matrix(matrix_dims, matrix_dims)};
    for (std::size_t row{0}; row < matrix_dims; ++row) {
        for (std::size_t col{0}; col < matrix_dims; col++) {
            // get cofactor of data[row][col]

            if (auto expected_minor{minor(data, row, col, matrix_dims)}; expected_minor.has_value()) {
                minor = std::move(expected_minor).value();
            }
            else {
                LOG(expected_minor.error());
                std::unreachable();
            }

            // sign of adj[col][row] positive if sum of row and column indexes is even
            if ((row + col) % 2 == 0) {
                sign = static_cast<value_type>(1);
            }
            else {
                sign = static_cast<value_type>(-1);
            }

            // complement is matrix of determinants of minors with alternating signs!!!
            if (auto expected_det{determinant(minor, matrix_dims - 1)}; expected_det.has_value()) {
                complement[row][col] = (sign)*std::move(expected_det).value();
            }
            else {
                LOG(expected_det.error());
                std::unreachable();
            }
        }
    }

    // adjostd::size_t is transposed of complement matrix
    return std::expected<matrix, matrix_error>{transposition(complement)};
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::inverse(const matrix& data) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    // matrix square
    const std::size_t matrix_dims{rows};
    // data is scalar, can omit later code
    if (matrix_dims == 1) {
        return std::expected<matrix, matrix_error>{data};
    }

    value_type det{};
    if (auto expected_det{determinant(data, matrix_dims)}; expected_det.has_value()) {
        det = std::move(expected_det).value();
    }
    else {
        LOG(expected_det.error());
        std::unreachable();
    }

    // assert correct determinant
    assert(det != 0);
    if (det == 0) {
        return std::unexpected<matrix_error>{matrix_error::singularity};
    }

    matrix adjoint{};
    if (auto expected_adjoint{adjoint(data)}; expected_adjoint.has_value()) {
        adjoint = std::move(expected_adjoint).value();
    }
    else {
        LOG(expected_adjoint.error());
        std::unreachable();
    }

    // inverse is adjoint matrix divided by det factor
    // division is multiplication by inverse
    return std::expected<matrix, matrix_error>{scale(adjoint, 1 / det)};
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::upper_triangular(const matrix& data) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    // matrix square
    const std::size_t matrix_dims{rows};
    // data is scalar
    if (matrix_dims == 1)
        return std::expected<matrix, matrix_error>{data};

    // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
    return std::expected<matrix, matrix_error>{transpose(lower_triangular(data))};
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::lower_triangular(const matrix& data) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};
    // assert correct dimensions
    assert(rows == cols);
    if (rows != cols) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    // matrix square
    const std::size_t matrix_dims = rows; // = cols;
    // data is scalar
    if (matrix_dims == 1) {
        return std::expected<matrix, matrix_error>{data};
    }

    matrix lower_triangular{make_matrix(matrix_dims, matrix_dims)};

    // decomposing data matrix into lower triangular
    for (std::size_t row{0}; row < matrix_dims; ++row) {
        for (std::size_t col{0}; col <= row; ++col) {
            value_type sum{};

            // summation for diagonals
            if (col == row) {
                for (std::size_t iSumCol{0}; iSumCol < col; ++iSumCol) {
                    sum += pow(lower_triangular[col][iSumCol], 2);
                }
                lower_triangular[col][col] = sqrt(data[col][col] - sum);
            }
            else {
                // evaluating L(row, col) using L(col, col)
                for (std::size_t iSumCol{0}; iSumCol < col; ++iSumCol) {
                    sum += (lower_triangular[row][iSumCol] * lower_triangular[col][iSumCol]);
                }
                lower_triangular[row][col] = (data[row][col] - sum) / lower_triangular[col][col];
            }
        }
    }
    return std::expected<matrix, matrix_error>{std::move(lower_triangular)};
}

template <typename value_type>
constexpr std::expected<matrix, matrix_error> matrix_wrapper<value_type>::product(const matrix& left,
                                                                                  const matrix& right) const
{
    const std::size_t left_rows{left.size()};
    const std::size_t right_rows{right.size()};
    const std::size_t left_cols = {left[0].size()};
    const std::size_t right_cols{right[0].size()};

    // assert correct dimensions
    assert(left_cols == right_rows);
    if (left_cols != right_rows) {
        return std::unexpected<matrix_error>{matrix_error::wrong_dims};
    }

    const std::size_t product_rows{left_rows};
    const std::size_t product_cols{right_cols};
    matrix product(product_rows, vector(product_cols, 0));

    for (std::size_t left_row{0}; left_row < left_rows; ++left_row) {
        for (std::size_t right_col{0}; right_col < right_cols; ++right_col) {
            value_type sum{0};
            for (std::size_t left_col{0}; left_col < left_cols; ++left_col) {
                sum += left[left_row][left_col] * right[left_col][right_col];
            }
            product[left_row][right_col] = sum;
        }
    }
    return std::expected<matrix, matrix_error>{std::move(product)};
}

template <typename value_type>
constexpr matrix matrix_wrapper<value_type>::scale(const matrix& data, const value_type factor) const
{
    const std::size_t rows{data.size()};
    const std::size_t cols{data[0].size()};

    // factor is 1 then dont need to do anything
    if (factor == 1) {
        return data;
    }
    // factor is 0 then return matrix of zeros
    else if (factor == 0) {
        data.clear();
        return data;
    }

    matrix scale{make_matrix(rows, cols)};
    for (std::size_t row{0}; row < rows; ++row) {
        for (std::size_t col{0}; col < cols; ++col) {
            scale[row][col] = data[row][col] * factor;
        }
    }
    return scale;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::append_row(const std::size_t row, const vector& new_row)
{
    assert(new_row.size() == columns_);
    data_.insert(std::next(data_.begin(), row), new_row); // basic pointer math (iterators are pointers)
    ++rows_;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::append_column(const std::size_t column, const vector& new_column)
{
    assert(new_column.size() == rows_);
    for (std::size_t row{0}; row < rows_; ++row) {
        data_[row].insert(std::next(data_[row].begin(), column),
                          new_column[row]); // basic pointer math (iterators are pointers)
    }
    ++columns_;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::delete_row(const std::size_t row)
{
    assert(row <= rows_);
    data_.erase(std::next(data_.begin(), row));
    --rows_;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::delete_column(const std::size_t column)
{
    assert(column <= columns_);
    for (std::size_t row{0}; row < rows_; ++row) {
        data_[row].erase(std::next(data_[row].begin(), column));
    }
    --columns_;
}

template <typename value_type>
constexpr vector matrix_wrapper<value_type>::end_row() const noexcept
{
    return data_.back();
}

template <typename value_type>
constexpr vector matrix_wrapper<value_type>::begin_row() const noexcept
{
    return data_.front();
}

template <typename value_type>
constexpr vector matrix_wrapper<value_type>::end_column() const
{
    vector end_column{};
    begin_column.reserve(rows_);
    for (std::size_t row{0}; row < rows_; ++row) {
        begin_column.push_back(data_[row].back());
    }
    return end_column;
}

template <typename value_type>
constexpr vector matrix_wrapper<value_type>::begin_column() const
{
    vector begin_column{};
    begin_column.reserve(rows_);
    for (std::size_t row{0}; row < rows_; ++row) {
        begin_column.push_back(data_[row].front());
    }
    return begin_column;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::resize(const std::size_t new_rows, const std::size_t new_cols)
{
    rows_ = new_rows;
    columns_ = new_cols;
    data_.resize(rows_);
    for (auto& row : data_) {
        row.resize(columns_);
    }
}

template <typename value_type>
constexpr inline void matrix_wrapper<value_type>::clear()
{
    data_.clear();
}

template <typename value_type>
constexpr inline constexpr bool matrix_wrapper<value_type>::empty() const noexcept
{
    return rows_ == columns_ == 0;
}

template <typename value_type>
constexpr std::size_t matrix_wrapper<value_type>::rows() const noexcept
{
    return rows_;
}

template <typename value_type>
constexpr std::size_t matrix_wrapper<value_type>::columns() const noexcept
{
    return columns_;
}

template <typename value_type>
constexpr vector matrix_wrapper<value_type>::diagonale() const
{
    assert(rows_ == columns_);

    vector diagonale(rows_);

    for (std::size_t diag{0}; diag < rows_; ++diag) {
        diagonale[diag] = data_[diag][diag];
    }
    return diagonale;
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::transpose() noexcept
{
    data_ = transposition(data_);

    // swap dimensions
    std::swap(rows_, columns_);
}

template <typename value_type>
constexpr void matrix_wrapper<value_type>::invert() noexcept
{
    if (auto inverse{inverse(data_)}; inverse.has_value()) {
        data_ = std::move(inverse).value();
    }
    else {
        LOG(inverse.error());
        std::unreachable();
    }
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator+(const matrix_wrapper& other) const noexcept
{
    // assert correct dimensions
    assert(other.rows_ == this->rows_);
    assert(other.columns_ == this->rows_);

    matrix_wrapper result{other.rows_, other.columns_};

    for (std::size_t row{0}; row < this->rows_; ++row) {
        for (std::size_t col{0}; col < this->columns_; ++col) {
            result.data_[row][col] = this->data_[row][col] + other.data_[row][col];
        }
    }
    return result;
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator+=(const matrix_wrapper& other) noexcept
{
    // assert correct dimensions
    assert(other.rows_ == this->rows_);
    assert(other.columns_ == this->rows_);

    for (std::size_t row{0}; row < this->rows_; ++row) {
        for (std::size_t col{0}; col < this->columns_; ++col) {
            this->data_[row][col] += other.data_[row][col];
        }
    }
    return *this;
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator-(const matrix_wrapper& other) const noexcept
{
    // assert correct dimensions
    assert(other.rows_ == this->rows_);
    assert(other.columns_ == this->rows_);

    matrix_wrapper result{other.rows_, other.columns_};

    for (std::size_t row{0}; row < this->rows_; ++row) {
        for (std::size_t col{0}; col < this->columns_; ++col) {
            result.data_[row][col] = this->data_[row][col] - other.data_[row][col];
        }
    }
    return result;
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator-=(const matrix_wrapper& other) noexcept
{
    // assert correct dimensions
    assert(other.rows_ == this->rows_);
    assert(other.columns_ == this->rows_);

    for (std::size_t row{0}; row < this->rows_; ++row) {
        for (std::size_t col{0}; col < this->columns_; ++col) {
            this->data_[row][col] -= other.data_[row][col];
        }
    }
    return *this;
}

template <typename value_type>
constexpr inline const matrix& matrix_wrapper<value_type>::data() const& noexcept
{
    return data_;
}

template <typename value_type>
constexpr inline matrix&& matrix_wrapper<value_type>::data() && noexcept
{
    return std::move(data_);
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator*(const value_type& factor) const noexcept
{
    // factor is 1 then dont need to do anything
    if (factor == 1) {
        return *this;
    }
    // factor is 0 then return matrix of zeros
    else if (factor == 0) {
        clear();
        return *this;
    }
    return matrix_wrapper{scale(data_, factor)}; // matrix_wrapper cstr no.1
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator*=(const value_type& factor) noexcept
{
    // factor is 1 then dont need to do anything
    if (factor == 1) {
        return *this;
    }
    // factor is 0 then return zero-ed this
    else if (factor == 0) {
        clear();
        return *this;
    }

    data_ = scale(data_, factor);
    return *this;
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator/(const value_type& factor) const noexcept
{
    // assert no division by 0!!!
    assert(factor != 0);

    // factor is 1 then dont need to do anything
    if (factor == 1) {
        return *this;
    }

    // division is multiplication by inverse
    return matrix_wrapper{scale(data_, 1 / factor)};
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator/=(const value_type& factor) noexcept
{
    // assert no division by 0!!!
    assert(factor != 0);

    // factor is 1 then dont need to do anything
    if (factor == 1) {
        return *this;
    }

    // division is multiplication by inverse
    data_ = scale(data_, 1 / factor);
    return *this;
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator*(const matrix_wrapper& other) const noexcept
{
    // assert correct dimensions
    assert(this->columns_ == other.rows_);

    result.rows_ = this->rows_;
    result.columns_ = other.columns_;
    return matrix_wrapper{product(this->data_, other.data_)};
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator*=(const matrix_wrapper& other) noexcept
{
    // assert correct dimensions
    assert(this->columns_ == other.rows_);

    this->data_ = product(this->data_, other.data_);
    this->rows_ = this->rows_;
    this->columns_ = other.columns_;
    return *this;
}

template <typename value_type>
constexpr matrix_wrapper matrix_wrapper<value_type>::operator/(const matrix_wrapper& other) const noexcept
{
    // assert correct dimensions
    assert(this->columns_ == other.rows_);

    // division is multiplication by inverse
    if (auto inverted{inverse(other.data_)}; inverted.has_value()) {
        if (auto product{product(data_, std::move(inverted).value())}; product.has_value()) {
            result.rows_ = this->rows_;
            result.columns_ = other.columns_;
            return matrix_wrapper{std::move(product).value()};
        }
        else {
            LOG(product.error());
            std::unreachable();
        }
    }
    else {
        LOG(inverted.error());
        std::unreachable();
    }
}

template <typename value_type>
constexpr matrix_wrapper& matrix_wrapper<value_type>::operator/=(const matrix_wrapper& other) noexcept
{
    // assert correct dimensions
    assert(this->columns_ == other.rows_);

    // division is multiplication by inverse
    if (auto inverted{inverse(other.data_)}; inverted.has_value()) {
        if (auto product{product(data_, std::move(inverted).value())}; product.has_value()) {
            *this = std::move(product).value();
            this->columns_ = other.columns_;
            return *this;
        }
        else {
            LOG(product.error());
            std::unreachable();
        }
    }
    else {
        LOG(inverted.error());
        std::unreachable();
    }
}

template <typename value_type>
constexpr inline void matrix_wrapper<value_type>::operator=(matrix&& data) noexcept
{
    data_ = std::forward<matrix>(data);
}

template <typename value_type>
constexpr inline void matrix_wrapper<value_type>::operator=(const matrix& data) noexcept
{
    data_ = data;
}

template <typename value_type>
constexpr const vector& matrix_wrapper<value_type>::operator()(std::size_t row) const noexcept
{
    return data_.at(row);
}
template <typename value_type>
constexpr const value_type& matrix_wrapper<value_type>::operator()(const std::size_t row,
                                                                   std::size_t column) const noexcept
{
    return data_.at(row)(column);
}

template <typename value_type>
constexpr vector& matrix_wrapper<value_type>::operator()(std::size_t row) noexcept
{
    return data_.at(row);
}

template <typename value_type>
constexpr value_type& matrix_wrapper<value_type>::operator()(const std::size_t row, std::size_t column) noexcept
{
    return data_.at(row)(column);
}

#endif // MATRIX_HPP