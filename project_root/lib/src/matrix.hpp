#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <cmath>
#include <expected>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mtx {

    enum class MatrixError { WrongDims, Singularity };

    inline static std::string_view matrixErrorToString(MatrixError error) {
        switch (error) {
            case MatrixError::WrongDims:
                return std::string_view{"Wrong dims"};
            case MatrixError::Singularity:
                return std::string_view{"Singularity"};
            default:
                return std::string_view{"None"};
        }
    }

    inline static void LOG(MatrixError error) {
        std::cerr << matrixErrorToString(error);
    }

    template <typename DataType>
    using vector = std::vector<DataType>;
    template <typename DataType>
    using matrix = std::vector<std::vector<DataType>>;

    template <typename DataType>
    class Matrix {
    public:
        explicit Matrix(size_t rows = 0, size_t cols = 0, DataType value = 0); // 0
        explicit Matrix(matrix<DataType>& otherData);                          // 1
        explicit Matrix(matrix<DataType> otherData);                           // 2
        explicit Matrix(const vector<DataType>& diag);                         // 3

        Matrix(const Matrix<DataType>&) = default;
        Matrix(Matrix<DataType>&&)      = default;
        Matrix<DataType>& operator=(const Matrix<DataType>&) noexcept;
        Matrix<DataType>& operator=(Matrix<DataType>&&) noexcept;
        ~Matrix() = default;

        void                    transpose();
        void                    invert();
        inline void             appendRow(size_t rowIdx, const vector<DataType>& row);
        inline void             appendColumn(size_t colIdx, const vector<DataType>& column);
        inline void             deleteRow(size_t rowIdx);
        inline void             deleteColumn(size_t colIdx);
        inline void             resize(size_t newRows, size_t newCols);
        inline size_t           getRows() const;
        inline size_t           getCols() const;
        inline vector<DataType> getEndRow() const;
        inline vector<DataType> getBeginRow() const;
        inline vector<DataType> getEndColumn() const;
        inline vector<DataType> getBeginColumn() const;
        inline vector<DataType> getDiag() const;

        inline Matrix<DataType>  operator*(const DataType& factor) const;
        inline Matrix<DataType>  operator/(const DataType& factor) const;
        inline Matrix<DataType>  operator+(const Matrix<DataType>& other) const;
        inline Matrix<DataType>  operator-(const Matrix<DataType>& other) const;
        inline Matrix<DataType>  operator*(const Matrix<DataType>& other) const;
        inline Matrix<DataType>  operator/(const Matrix<DataType>& other) const;
        inline Matrix<DataType>& operator*=(const DataType& factor);
        inline Matrix<DataType>& operator/=(const DataType& factor);
        inline Matrix<DataType>& operator+=(const Matrix<DataType>& other);
        inline Matrix<DataType>& operator-=(const Matrix<DataType>& other);
        inline Matrix<DataType>& operator*=(const Matrix<DataType>& other);
        inline Matrix<DataType>& operator/=(const Matrix<DataType>& other);
        inline vector<DataType>  operator()(size_t rowIdx) const;
        inline DataType          operator()(size_t rowIdx, size_t colIdx) const;

    private:
        std::expected<DataType, MatrixError> getDeterminant(const matrix<DataType>& data, size_t matrixDims) const;
        std::expected<matrix<DataType>, MatrixError> getMinor(const matrix<DataType>& data, size_t rowIdx,
                                                              size_t colIdx, size_t matrixDims) const;
        matrix<DataType>                             getTranspose(const matrix<DataType>& data) const;
        std::expected<matrix<DataType>, MatrixError> getAdjoint(const matrix<DataType>& data) const;
        std::expected<matrix<DataType>, MatrixError> getInverse(const matrix<DataType>& data) const;
        std::expected<matrix<DataType>, MatrixError> getLowerTriangular(const matrix<DataType>& data) const;
        std::expected<matrix<DataType>, MatrixError> getUpperTriangular(const matrix<DataType>& data) const;
        matrix<DataType> getScale(const matrix<DataType>& data, const DataType& factor) const;
        std::expected<matrix<DataType>, MatrixError> getProduct(const matrix<DataType>& left,
                                                                const matrix<DataType>& right) const;

        size_t           rows_{}; // first dim
        size_t           cols_{}; // second dim
        DataType         det_{};
        matrix<DataType> data_{}; // vector of vectors
    };

    template <typename DataType>
    Matrix<DataType>::Matrix(size_t rows, size_t cols, DataType value) :
        rows_(rows), cols_(cols), data_(rows_, vector<DataType>(cols_, value)) {
        // assert that either both dimensions were specified, or neither
        assert((rows == 0 && cols == 0) || (rows != 0 && cols != 0));
    }

    template <typename DataType>
    Matrix<DataType>::Matrix(const vector<DataType>& diag) :
        rows_(diag.size()), cols_(rows_), data_(rows_, vector<DataType>(cols_, 0)) {
        // create diagonal matrix with zeros beside diagonale
        for (size_t row{0}; row < rows_; ++row) {
            for (size_t col{0}; col < cols_; ++col) {
                if (row == col) {
                    data_[row][col] = diag[row];
                }
            }
        }
    }

    template <typename DataType>
    Matrix<DataType>::Matrix(matrix<DataType> otherData) :
        data_(std::move(otherData)), rows_(otherData.size()), cols_(otherData[0].size()) {
    }

    template <typename DataType>
    Matrix<DataType>::Matrix(matrix<DataType>& otherData) :
        data_(std::move(otherData)), rows_(otherData.size()), cols_(otherData[0].size()) {
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError> Matrix<DataType>::getMinor(const matrix<DataType>& data, size_t rowIdx,
                                                                            size_t colIdx, size_t matrixDims) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }
        // assert cofactor isnt calculated for minor bigger than data
        assert(matrixDims < rowIdx && matrixDims < colIdx);
        if (matrixDims >= rowIdx || matrixDims >= colIdx) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }
        // minor is scalar, can omit later code
        if (matrixDims == 0) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place, data};
        }

        matrix<DataType> minor(matrixDims, vector<DataType>(matrixDims, 0));
        size_t           cofRow{0};
        size_t           cofCol{0};
        for (size_t dataRow{0}; dataRow < matrixDims; ++dataRow) {
            for (size_t dataCol{0}; dataCol < matrixDims; ++dataCol) {
                // copying into cofactor matrix only those element which are not in given row and column
                if (dataRow != rowIdx && dataCol != colIdx) {
                    minor[cofRow][cofCol++] = data[dataRow][dataCol];

                    // row is filled, so increase row index and reset col index
                    if (cofCol == matrixDims - 1) {
                        cofCol = 0;
                        ++cofRow;
                    }
                }
            }
        }
        return std::expected<matrix<DataType>, MatrixError>{std::in_place, minor};
    }

    template <typename DataType>
    std::expected<DataType, MatrixError> Matrix<DataType>::getDeterminant(const matrix<DataType>& data,
                                                                          size_t                  matrixDims) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        // assert minor isnt bigger than data
        assert(rows > matrixDims && cols > matrixDims);
        if (rows <= matrixDims || cols <= matrixDims) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError: WrongDims};
        }

        // data is scalar, can omit later code
        if (matrixDims == 1) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place, data[0][0]};
        }
        // data is 2x2 matrix, can omit later code
        if (matrixDims == 2) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place,
                                                                (data[0][0] * data[1][1]) - (data[1][0] * data[0][1])};
        }

        // result object
        DataType det{static_cast<DataType>(0)};

        // sign multiplier
        DataType         sign = static_cast<DataType>(1);
        matrix<DataType> minor(matrixDims, vector<DataType>(cols, matrixDims));
        for (size_t col{0}; col < matrixDims; ++col) {
            // cofactor of data[0][col]

            // moving temporary prevents copy elision, which is even better
            // if (auto expected{std::move(getMinor(data, 0, col, matrixDims))}; expected.has_value()) {
            //     minor = std::move(expected.value());
            // } else {
            //     LOG(expected.error());
            //     std::abort();
            // }
            if (auto expected{getMinor(data, 0, col, matrixDims)}; expected.has_value()) {
                minor = expected.value();
            } else {
                LOG(expected.error());
                std::abort();
            }

            det += sign * data[0][col] * getDeterminant(minor, matrixDims - 1);
            // alternate sign
            sign *= static_cast<DataType>(-1);
        }

        return std::expected<matrix<DataType>, MatrixError>{std::in_place, det};
    }

    template <typename DataType>
    matrix<DataType> Matrix<DataType>::getTranspose(const matrix<DataType>& data) const {
        size_t newRows{data.size()};
        size_t newCols{data[0].size()};

        // data is scalar, can omit later code
        if ((newRows == newCols) == 1)
            return data;

        // result object
        matrix<DataType> transpose(newRows, vector<DataType>(newCols, 0));

        for (size_t row{0}; row < newRows; ++row) {
            for (size_t col{0}; col < newCols; ++col) {
                transpose[row][col] = data[col][row];
            }
        }
        return transpose;
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError> Matrix<DataType>::getAdjoint(const matrix<DataType>& data) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        // matrix square
        size_t matrixDims{rows};
        // data is scalar, can omit later code
        if (matrixDims == 1) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place, data};
        }

        matrix<DataType> complement(matrixDims, vector<DataType>(matrixDims, 0));

        // sign multiplier
        DataType         sign{static_cast<DataType>(1)};
        matrix<DataType> minor(matrixDims, vector<DataType>(matrixDims, 0));
        for (size_t row{0}; row < matrixDims; ++row) {
            for (size_t col{0}; col < matrixDims; col++) {
                // get cofactor of data[row][col]

                // moving temporary prevents copy elision, which is even better
                // if (auto expected{std::move(getMinor(data, row, col, matrixDims))}; expected.has_value()) {
                //     minor = std::move(expected.value());
                // } else {
                //     LOG(expected.error());
                //     std::abort();
                // }

                if (auto expected{getMinor(data, row, col, matrixDims)}; expected.has_value()) {
                    minor = expected.value();
                } else {
                    LOG(expected.error());
                    std::abort();
                }

                // sign of adj[col][row] positive if sum of row and column indexes is even
                if ((row + col) % 2 == 0) {
                    sign = static_cast<DataType>(1);
                } else {
                    sign = static_cast<DataType>(-1);
                }

                // complement is matrix of determinants of minors with alternating signs!!!
                complement[row][col] = (sign) * (getDeterminant(minor, matrixDims - 1));
            }
        }
        // result object
        // adjosize_t is transposed of complement matrix
        return std::expected<matrix<DataType>, MatrixError>{std::in_place,
                                                            matrix<DataType>{std::move(getTranspose(complement))}};
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError> Matrix<DataType>::getInverse(const matrix<DataType>& data) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        // matrix square
        size_t matrixDims{rows};
        // data is scalar, can omit later code
        if (matrixDims == 1) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place, data};
        }

        DataType det{getDeterminant(data, matrixDims)};
        // assert correct determinant
        assert(det != 0);
        if (det == 0) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::Singularity};
        }

        matrix<DataType> adjoint{std::move(getAdjoint(data))};

        // inverse is adjoint matrix divided by det factor
        // division is multiplication by inverse
        return std::expected<matrix<DataType>, MatrixError>{std::in_place,
                                                            matrix<DataType>{std::move(getScale(adjoint, 1 / det))}};
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError>
    Matrix<DataType>::getUpperTriangular(const matrix<DataType>& data) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1)
            return data;

        // result object
        // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
        return std::expected<matrix<DataType>, MatrixError>{
            std::in_place, matrix<DataType>(std::move(getTranspose(getLowerTriangular(data))))};
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError>
    Matrix<DataType>::getLowerTriangular(const matrix<DataType>& data) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};
        // assert correct dimensions
        assert(rows == cols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1) {
            return std::expected<matrix<DataType>, MatrixError>{std::in_place, data};
        }

        // result object
        matrix<DataType> lowerTriangular(matrixDims, vector<DataType>(matrixDims, 0));

        // decomposing data matrix into lower triangular
        for (size_t row{0}; row < matrixDims; ++row) {
            for (size_t col{0}; col <= row; ++col) {
                DataType sum{0};

                // summation for diagonals
                if (col == row) {
                    for (size_t iSumCol{0}; iSumCol < col; ++iSumCol) {
                        sum += pow(lowerTriangular[col][iSumCol], 2);
                    }
                    lowerTriangular[col][col] = sqrt(data[col][col] - sum);
                } else {
                    // evaluating L(row, col) using L(col, col)
                    for (size_t iSumCol{0}; iSumCol < col; ++iSumCol) {
                        sum += (lowerTriangular[row][iSumCol] * lowerTriangular[col][iSumCol]);
                    }
                    lowerTriangular[row][col] = (data[row][col] - sum) / lowerTriangular[col][col];
                }
            }
        }
        return std::expected<matrix<DataType>, MatrixError>{std::in_place, lowerTriangular};
    }

    template <typename DataType>
    std::expected<matrix<DataType>, MatrixError> Matrix<DataType>::getProduct(const matrix<DataType>& left,
                                                                              const matrix<DataType>& right) const {
        size_t leftRows = left.size(), rightRows = right.size();
        size_t leftCols = left[0].size(), rightCols = right[0].size();

        // assert correct dimensions
        assert(leftCols == rightCols);
        if (rows != cols) {
            return std::unexpected<MatrixError>{std::in_place, MatrixError::WrongDims};
        }

        size_t           productRows = leftRows;
        size_t           productCols = rightCols;
        matrix<DataType> product(productRows, vector<DataType>(productCols, 0));

        for (size_t leftRow{0}; leftRow < leftRows; ++leftRow) {
            for (size_t rightCol{0}; rightCol < rightCols; ++rightCol) {
                DataType sum{0};
                for (size_t leftCol{0}; leftCol < leftCols; ++leftCol) {
                    sum += left[leftRow][leftCol] * right[leftCol][rightCol];
                }
                product[leftRow][rightCol] = sum;
            }
        }
        return std::expected<matrix<DataType>, MatrixError>{std::in_place, product};
    }

    template <typename DataType>
    matrix<DataType> Matrix<DataType>::getScale(const matrix<DataType>& data, const DataType& factor) const {
        size_t rows{data.size()};
        size_t cols{data[0].size()};

        // factor is 1 then dont need to do anything
        if (factor == 1)
            return data;
        // factor is 0 then return matrix of zeros
        else if (factor == 0)
            return matrix<DataType>(rows, vector<DataType>(cols, 0));

        matrix<DataType> scale(rows, vector<DataType>(rows, 0));

        for (size_t row{0}; row < rows; ++row) {
            for (size_t col{0}; col < cols; ++col) {
                scale[row][col] = data_[row][col] * factor;
            }
        }
        return scale;
    }

    template <typename DataType>
    void Matrix<DataType>::appendRow(size_t rowIdx, const vector<DataType>& row) {
        assert(row.size() == cols_);
        data_.insert(data_.begin() + rowIdx, row); // basic pointer math (iterators are pointers)
        ++rows_;
    }

    template <typename DataType>
    void Matrix<DataType>::appendColumn(size_t colIdx, const vector<DataType>& column) {
        assert(column.size() == rows_);
        for (size_t row{0}; row < rows_; ++row) {
            data_[row].insert(data_[row].begin() + colIdx, column[row]); // basic pointer math (iterators are pointers)
        }
        ++cols_;
    }

    template <typename DataType>
    void Matrix<DataType>::deleteRow(size_t rowIdx) {
        assert(rowIdx <= rows_);
        data_.erase(data_.begin() + rowIdx); // basic pointer math (iterators are pointers)
        --rows_;
    }

    template <typename DataType>
    void Matrix<DataType>::deleteColumn(size_t colIdx) {
        assert(colIdx <= cols_);
        for (size_t row{0}; row < rows_; ++row) {
            data_[row].erase(data_[row].begin() + colIdx); // basic pointer math (iterators are pointers)
        }
        --cols_;
    }

    template <typename DataType>
    vector<DataType> Matrix<DataType>::getEndRow() const {
        return *data_.back();
    }

    template <typename DataType>
    vector<DataType> Matrix<DataType>::getBeginRow() const {
        return *data_.begin();
    }

    template <typename DataType>
    vector<DataType> Matrix<DataType>::getEndColumn() const {
        vector<DataType> ret(rows_, 0);
        for (size_t row{0}; row < rows_; ++row) {
            ret.emplace_back(*data_[row].back());
        }
        return ret;
    }

    template <typename DataType>
    vector<DataType> Matrix<DataType>::getBeginColumn() const {
        vector<DataType> ret(rows_, 0);
        for (size_t row{0}; row < rows_; ++row) {
            ret.emplace_back(*data_[row].end());
        }
        return ret;
    }

    template <typename DataType>
    inline void Matrix<DataType>::resize(size_t newRows, size_t newCols) {
        assert(newRows > 0);
        assert(newCols > 0);

        rows_ = newRows;
        cols_ = newCols;
        data_.resize(rows_);
        for (auto& row : data_) {
            row.resize(cols_);
        }
    }

    template <typename DataType>
    inline size_t Matrix<DataType>::getRows() const {
        return rows_;
    }

    template <typename DataType>
    inline size_t Matrix<DataType>::getCols() const {
        return cols_;
    }

    template <typename DataType>
    vector<DataType> Matrix<DataType>::getDiag() const {
        // throw exception for wrong dims

        vector<DataType> ret(rows_, 0);

        for (size_t diag{0}; diag < rows_; ++diag) {
            ret[diag] = data_[diag][diag];
        }
        return ret;
    }

    template <typename DataType>
    void Matrix<DataType>::transpose() {
        data_ = std::move(getTranspose(data_));

        // swap dimensions
        std::swap(rows_, cols_);
    }

    template <typename DataType>
    void Matrix<DataType>::invert() {
        data_ = std::move(getInverse(data_));
    }

    // addition operator (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator+(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(other.rows_ == rows_);
        assert(other.cols_ == rows_);

        Matrix<DataType> result(other.rows_, other.cols_);

        for (size_t row{0}; row < rows_; ++row) {
            for (size_t col{0}; col < cols_; ++col) {
                result.data_[row][col] = data_[row][col] + other.data_[row][col];
            }
        }
        return result;
    }

    // addition operator (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator+=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(other.rows_ == rows_);
        assert(other.cols_ == rows_);

        for (size_t row{0}; row < rows_; ++row) {
            for (size_t col{0}; col < cols_; ++col) {
                data_[row][col] += other.data_[row][col];
            }
        }
        return *this;
    }

    // substraction operator (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator-(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(other.rows_ == rows_);
        assert(other.cols_ == rows_);

        Matrix<DataType> result(other.rows_, other.cols_);

        for (size_t row{0}; row < rows_; ++row) {
            for (size_t col{0}; col < cols_; ++col) {
                result.data_[row][col] = data_[row][col] - other.data_[row][col];
            }
        }
        return result;
    }

    // substraction operator (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator-=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(other.rows_ == rows_);
        assert(other.cols_ == rows_);

        for (size_t row{0}; row < rows_; ++row) {
            for (size_t col{0}; col < cols_; ++col) {
                data_[row][col] -= other.data_[row][col];
            }
        }
        return *this;
    }

    // multiplication operator for scalars (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator*(const DataType& factor) const {
        // factor is 1 then dont need to do anything
        if (factor == 1)
            return *this;
        // factor is 0 then return matrix of zeros
        else if (factor == 0)
            return Matrix<DataType>(rows_, cols_, 0);

        return Matrix<DataType>(std::move(getScale(data_, factor))); // Matrix cstr no.1
    }

    // multiplication operator for scalars (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator*=(const DataType& factor) {
        // factor is 1 then dont need to do anything
        if (factor == 1)
            return *this;
        // factor is 0 then return zero-ed this
        else if (factor == 0) {
            *this = Matrix<DataType>(rows_, cols_, 0); // Matrix move operator (*this is already constructed)
            return *this;
        }

        data_ = std::move(getScale(data_, factor)); // vector move operator
        return *this;
    }

    // division operator for scalars (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator/(const DataType& factor) const {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1)
            return *this;

        // division is multiplication by inverse
        return Matrix<DataType>(std::move(getScale(data_, 1 / factor))); // Matrix cstr no.1
    }

    // division operator for scalars (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator/=(const DataType& factor) {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1)
            return *this;

        // division is multiplication by inverse
        data_ = std::move(getScale(data_, 1 / factor)); // vector move operator
        return *this;
    }

    // multiplication operator for matrixes (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator*(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(cols_ == other.rows_);

        Matrix<DataType> result(std::move(getProduct(data_, other.data_))); // Matrix cstr no.1

        result.rows_ = rows_;
        result.cols_ = other.cols_;
        return result;
    }

    // multiplication operator for matrixes (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator*=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(cols_ == other.rows_);

        data_ = std::move(getProduct(data_, other.data_));

        rows_ = rows_;
        cols_ = other.cols_;
        return *this;
    }

    // division operator for matrixes (return copy of result), using multiplication operator
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator/(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(cols_ == other.rows_);

        if (auto expected{std::move(getProduct(data_, getInverse(other.data_)))}; expected.has_value()) {
            Matrix<DataType> result{std::move(expected.value())};
            result.resize(rows_, other.cols_);
            return result;
        } else {
            LOG(expected.error());
            std::abort();
        }
    }

    // division operator for matrixes (return reference to this (lhs)), using multiplication operator
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator/=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(cols_ == other.rows_);

        // division is multiplication by inverse
        if (auto expected{std::move(getProduct(data_, getInverse(other.data_)))}; expected.has_value()) {
            Matrix<DataType> result{std::move(expected.value())};
            result.resize(rows_, other.cols_);
            return result;
        } else {
            LOG(expected.error());
            std::abort();
        }

        this->resize(rows_, other.cols_);
        return *this;
    }

    // copy operator (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator=(const Matrix<DataType>& other) noexcept {
        // same container inside objects, no need to move entire object
        if (data_ == other.data_)
            return *this;

        data_ = other.data_;
        cols_ = other.cols_;
        rows_ = other.rows_;
        return *this;
    }

    // move operator (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator=(Matrix<DataType>&& other) noexcept {
        // same container inside objects, no need to move entire object
        if (data_ == other.data_)
            return *this;

        data_ = std::move(other.data_);
        // no move operator for trivial
        cols_ = other.cols_;
        rows_ = other.rows_;
        return *this;
    }

    // index operator for rows (return reference of result)
    template <typename DataType>
    inline vector<DataType> Matrix<DataType>::operator()(size_t rowIdx) const {
        return data_[rowIdx];
    }

    // index operator for rows/cols (elements) (return reference of result)
    template <typename DataType>
    inline DataType Matrix<DataType>::operator()(size_t rowIdx, size_t colIdx) const {
        return data_[rowIdx][colIdx];
    }

} // namespace mtx

#endif // MATRIX_HPP

int main() {
    mtx::Matrix<float> matrix{1, 1, 2};

    return 0;
}
