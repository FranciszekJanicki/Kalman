#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "config.h"

namespace mtx {

    template <typename DataType>
    class Matrix {
    public:
        explicit Matrix(size_t rows = 0, size_t cols = 0, DataType value = 0); // 0
        Matrix(std::vector<std::vector<DataType>> otherData);                  // 1
        Matrix(const std::vector<DataType>& diag);                             // 2
        Matrix(const Matrix<DataType>&) = default;
        Matrix(Matrix<DataType>&&)      = default;
        Matrix<DataType>& operator=(const Matrix<DataType>&);
        Matrix<DataType>& operator=(Matrix<DataType>&&) noexcept;
        ~Matrix() = default;

        void                         transpose();
        void                         invert();
        inline void                  appendRow(size_t rowIdx, const std::vector<DataType>& row);
        inline void                  appendColumn(size_t colIdx, const std::vector<DataType>& column);
        inline void                  deleteRow(size_t rowIdx);
        inline void                  deleteColumn(size_t colIdx);
        inline void                  resize(size_t newRows, size_t newCols);
        inline size_t                getRows() const;
        inline size_t                getCols() const;
        inline std::vector<DataType> getEndRow() const;
        inline std::vector<DataType> getBeginRow() const;
        inline std::vector<DataType> getEndColumn() const;
        inline std::vector<DataType> getBeginColumn() const;
        inline std::vector<DataType> getDiag() const;

        inline Matrix<DataType>      operator*(const DataType& factor) const;
        inline Matrix<DataType>      operator/(const DataType& factor) const;
        inline Matrix<DataType>      operator+(const Matrix<DataType>& other) const;
        inline Matrix<DataType>      operator-(const Matrix<DataType>& other) const;
        inline Matrix<DataType>      operator*(const Matrix<DataType>& other) const;
        inline Matrix<DataType>      operator/(const Matrix<DataType>& other) const;
        inline Matrix<DataType>&     operator*=(const DataType& factor);
        inline Matrix<DataType>&     operator/=(const DataType& factor);
        inline Matrix<DataType>&     operator+=(const Matrix<DataType>& other);
        inline Matrix<DataType>&     operator-=(const Matrix<DataType>& other);
        inline Matrix<DataType>&     operator*=(const Matrix<DataType>& other);
        inline Matrix<DataType>&     operator/=(const Matrix<DataType>& other);
        inline std::vector<DataType> operator()(size_t rowIdx) const;
        inline DataType              operator()(size_t rowIdx, size_t colIdx) const;

    private:
        // PRIVATE FUNCTION MEMBERS
        DataType                           getDeterminant(const std::vector<std::vector<DataType>>& data, size_t matrixDims) const;
        std::vector<std::vector<DataType>> getMinor(const std::vector<std::vector<DataType>>& data, size_t rowIdx, size_t colIdx,
                                                    size_t matrixDims) const;
        std::vector<std::vector<DataType>> getTranspose(const std::vector<std::vector<DataType>>& data) const;
        std::vector<std::vector<DataType>> getAdjoint(const std::vector<std::vector<DataType>>& data) const;
        std::vector<std::vector<DataType>> getInverse(const std::vector<std::vector<DataType>>& data) const;
        std::vector<std::vector<DataType>> getLowerTriangular(const std::vector<std::vector<DataType>>& data) const;
        std::vector<std::vector<DataType>> getUpperTriangular(const std::vector<std::vector<DataType>>& data) const;
        std::vector<std::vector<DataType>> getScale(const std::vector<std::vector<DataType>>& data, const DataType& factor) const;
        std::vector<std::vector<DataType>> getProduct(const std::vector<std::vector<DataType>>& left,
                                                      const std::vector<std::vector<DataType>>& right) const;

        // DATA MEMBERS
        size_t                             rows_{}; // first dim
        size_t                             cols_{}; // second dim
        DataType                           det_{};
        std::vector<std::vector<DataType>> data_{}; // vector of vectors
    };

    template <typename DataType>
    Matrix<DataType>::Matrix(size_t rows, size_t cols, DataType value) :
        rows_(rows), cols_(cols), data_(rows_, std::vector<DataType>(cols_, value)) {
        // assert that either both dimensions were specified, or neither
        assert((rows == 0 && cols == 0) || (rows != 0 && cols != 0));

        // for (int iRow = 0; iRow < rows_; ++iRow) {
        //     data_.emplace_back(std::vector<DataType>); // emplace_back taking rvalue
        //     for (int iCol = 0; iCol < cols_; ++iCol) {
        //         // data_[iRow].emplace_back(value);
        //         data_.end()->emplace_back(value); // emplace_back taking rvalue
        //     }
        // }
    }

    template <typename DataType>
    Matrix<DataType>::Matrix(const std::vector<DataType>& diag) :
        rows_(diag.size()), cols_(rows_), data_(rows_, std::vector<DataType>(cols_, 0)) {
        // create diagonal matrix with zeros beside diagonale
        for (int iRow = 0; iRow < rows_; ++iRow) {
            for (int iCol = 0; iCol < cols_; ++iCol) {
                if (iRow == iCol) {
                    data_[iRow][iCol] = diag[iRow];
                }
            }
        }
    }
    // template values specified during compile time (object declaration),
    // can use move constructor during runtime, to initialize allocataed data_ memory
    template <typename DataType> // (othetData is passed by rvalue ref, so its move cstr)
    Matrix<DataType>::Matrix(std::vector<std::vector<DataType>> otherData) :
        data_(std::move(otherData)), rows_(otherData.size()), cols_(otherData[0].size()) {
    }

    // if you dont need a custom destructor, then don't, as creating your own destructors (aswell as copy constructors) can disable the
    // compilers ability to optimize assembly instructions using move operations template <typename DataType> Matrix<DataType>::~Matrix() {
    //     // compiler will use default (most optimized) self created destructor, std::vector will take care of its memory on the heap
    // }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getMinor(const std::vector<std::vector<DataType>>& data, size_t rowIdx,
                                                                  size_t colIdx, size_t matrixDims) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
#if USE_EXCEPTIONS
        if (rows != cols)
            throw std::runtime_error("No cofactor for non-square matrix!");
#endif

        // assert cofactor isnt calculated for minor bigger than data
        assert(matrixDims < rowIdx && matrixDims < colIdx);
#if USE_EXCEPTIONS
        if (matrixDims >= rowIdx || matrixDims >= colIdx)
            throw std::runtime_error("Wrong minor!");
#endif

        // minor is scalar, can omit later code
        if (matrixDims == 0)
            return data;

        // result object
        std::vector<std::vector<DataType>> minor(matrixDims, std::vector<DataType>(matrixDims, 0));

        size_t iCofRow = 0, iCofCol = 0;
        for (size_t iDataRow = 0; iDataRow < matrixDims; ++iDataRow) {
            for (size_t iDataCol = 0; iDataCol < matrixDims; ++iDataCol) {
                // copying into cofactor matrix only those element which are not in given row and column
                if (iDataRow != rowIdx && iDataCol != colIdx) {
                    minor[iCofRow][iCofCol++] = data[iDataRow][iDataCol];

                    // row is filled, so increase row index and reset col index
                    if (iCofCol == matrixDims - 1) {
                        iCofCol = 0;
                        ++iCofRow;
                    }
                }
            }
        }
        return minor;
    }

    template <typename DataType>
    DataType Matrix<DataType>::getDeterminant(const std::vector<std::vector<DataType>>& data, size_t matrixDims) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
#if USE_EXCEPTIONS
        if (rows != cols)
            throw std::runtime_error("No cofactor for non-square matrix!");
#endif

        // assert minor isnt bigger than data
        assert(matrixDims < rows && matrixDims < cols);
#if USE_EXCEPTIONS
        if (matrixDims >= rows || matrixDims >= cols)
            throw std::runtime_error("Wrong data!");
#endif

        // data is scalar, can omit later code
        if (matrixDims == 1) {
            return data[0][0];
        }
        // data is 2x2 matrix, can omit later code
        if (matrixDims == 2) {
            return ((data[0][0] * data[1][1]) - (data[1][0] * data[0][1]));
        }

        // result object
        DataType det = 0;

        // sign multiplier
        DataType                           sign = 1;
        std::vector<std::vector<DataType>> minor(matrixDims, std::vector<DataType>(cols, matrixDims));
        for (size_t iCol = 0; iCol < matrixDims; ++iCol) {
// cofactor of data[0][iCol]
#if USE_EXCEPTIONS
            try {
                minor = std::move(getMinor(data, 0, iCol, matrixDims)); // std::vector move operator
            } catch (const std::runtime_error& error) {
                LOG(error);
            }

            try {
                det += sign * data[0][iCol] * determinant(minor, matrixDims - 1);
            } catch (const std::runtime_error& error) {
                LOG(error);
            }
#else
            minor = std::move(getMinor(data, 0, iCol, matrixDims)); // std::vector move operator
            det += sign * data[0][iCol] * getDeterminant(minor, matrixDims - 1);
#endif

            // alternate sign
            sign *= -1;
        }

        return det;
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getTranspose(const std::vector<std::vector<DataType>>& data) const {
        size_t newRows = data[0].size();
        size_t newCols = data.size();

        // data is scalar, can omit later code
        if (newRows == newCols == 1)
            return data;

        // result object
        std::vector<std::vector<DataType>> transpose(newRows, std::vector<DataType>(newCols, 0));

        for (int iRow = 0; iRow < newRows; ++iRow) {
            for (int iCol = 0; iCol < newCols; ++iCol) {
                transpose[iRow][iCol] = data[iCol][iRow];
            }
        }
        return transpose;
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getAdjoint(const std::vector<std::vector<DataType>>& data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
// throw exception if non-square
#if USE_EXCEPTIONS
        if (rows != cols)
            throw std::runtime_error("No adjoint for non-square matrix!");
#endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar, can omit later code
        if (matrixDims == 1)
            return data;

        std::vector<std::vector<DataType>> complement(matrixDims, std::vector<DataType>(matrixDims, 0));

        // sign multiplier
        DataType                           sign = 1;
        std::vector<std::vector<DataType>> minor(matrixDims, std::vector<DataType>(matrixDims, 0));
        for (int iRow = 0; iRow < matrixDims; ++iRow) {
            for (int iCol = 0; iCol < matrixDims; iCol++) {
// get cofactor of data[iRow][iCol]
#if USE_EXCEPTIONS
                try {
                    minor = std::move(getMinor(data, iRow, iCol, matrixDims));
                } catch (const std::runtime_error& error) {
                    LOG(error);
                }
                // sign of adj[iCol][iRow] positive if sum of row and column indexes is even
                if ((iRow + iCol) % 2 == 0)
                    sign = 1;
                else
                    sign = -1;

                // complement is matrix of determinants of minors with alternating signs!!!
                try {
                    complement[iRow][iCol] = (sign) * (determinant(minor, matrixDims - 1));
                } catch (const std::runtime_error& error) {
                    LOG(error);
                }
#else
                minor = std::move(getMinor(data, iRow, iCol, matrixDims));
                // sign of adj[iCol][iRow] positive if sum of row and column indexes is even
                if ((iRow + iCol) % 2 == 0)
                    sign = 1;
                else
                    sign = -1;

                // complement is matrix of determinants of minors with alternating signs!!!
                complement[iRow][iCol] = (sign) * (getDeterminant(minor, matrixDims - 1));
#endif
            }
        }
        // result object
        // adjoint is transposed of complement matrix
        return std::vector<std::vector<DataType>>(
            std::move(getTranspose(complement))); // dont declare if all you need is to return or copy it into collection, instead just use
                                                  // constructor without declaring varaible if it isnt used
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getInverse(const std::vector<std::vector<DataType>>& data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
/// throw exception if non-square
#if USE_EXCEPTIONS
        if (rows != cols)
            throw std::runtime_error("No inverse for non-square matrix!");
#endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar, can omit later code
        if (matrixDims == 1)
            return data;

        DataType det = getDeterminant(data, matrixDims);
        // assert correct determinant
        assert(det != 0);
// throw exception if singular
#if USE_EXCEPTIONS
        if (det == 0)
            throw std::runtime_error("No inverse for singular matrix!");
#endif

#if USE_EXCEPTIONS
        try {
            std::vector<std::vector<DataType>> adjoint(std::move(getAdjoint(data)));
        } catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        std::vector<std::vector<DataType>> adjoint(std::move(getAdjoint(data)));
#endif

        // inverse is adjoint matrix divided by det factor
        // division is multiplication by inverse
        return std::vector<std::vector<DataType>>(std::move(getScale(adjoint, 1 / det)));
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getUpperTriangular(const std::vector<std::vector<DataType>>& data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
// throw exception if non-square
#if USE_EXCEPTIONS
        if (cols != rows)
            throw std::runtime_error("No upper-triangular for non-square matrix!");
#endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1)
            return data;

        // result object
        // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
        return std::vector<std::vector<DataType>>(std::move(getTranspose(getLowerTriangular(data))));
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getLowerTriangular(const std::vector<std::vector<DataType>>& data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions
        assert(rows == cols);
// throw exception if non-square
#if USE_EXCEPTIONS
        if (cols != rows)
            throw std::runtime_error("No lower-triangular for non-square matrix!");
#endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1)
            return data;

        // result object
        std::vector<std::vector<DataType>> lowerTriangular(matrixDims, std::vector<DataType>(matrixDims, 0));

        // decomposing data matrix into lower triangular
        for (int iRow = 0; iRow < matrixDims; ++iRow) {
            for (int iCol = 0; iCol <= iRow; ++iCol) {
                DataType sum = 0;

                // summation for diagonals
                if (iCol == iRow) {
                    for (int iSumCol = 0; iSumCol < iCol; ++iSumCol) {
                        sum += pow(lowerTriangular[iCol][iSumCol], 2);
                    }
                    lowerTriangular[iCol][iCol] = sqrt(data[iCol][iCol] - sum);
                } else {
                    // evaluating L(iRow, iCol) using L(iCol, iCol)
                    for (int iSumCol = 0; iSumCol < iCol; ++iSumCol) {
                        sum += (lowerTriangular[iRow][iSumCol] * lowerTriangular[iCol][iSumCol]);
                    }
                    lowerTriangular[iRow][iCol] = (data[iRow][iCol] - sum) / lowerTriangular[iCol][iCol];
                }
            }
        }
        return lowerTriangular;
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getProduct(const std::vector<std::vector<DataType>>& left,
                                                                    const std::vector<std::vector<DataType>>& right) const {
        size_t leftRows = left.size(), rightRows = right.size();
        size_t leftCols = left[0].size(), rightCols = right[0].size();

        // assert correct dimensions
        assert(leftCols == rightCols);
// throw exception to incorrect dimensions
#if USE_EXCEPTIONS
        if (leftCols != rightCols)
            throw std::runtime_error("No cholseky for non-square matrix!");
#endif

        size_t                             productRows = leftRows;
        size_t                             productCols = rightCols;
        std::vector<std::vector<DataType>> product(productRows, std::vector<DataType>(productCols, 0));

        for (size_t iLeftRow = 0; iLeftRow < leftRows; ++iLeftRow) {
            for (size_t iRightCol = 0; iRightCol < rightCols; ++iRightCol) {
                DataType sum = 0;
                for (size_t iLeftCol = 0; iLeftCol < leftCols; ++iLeftCol) {
                    sum += left[iLeftRow][iLeftCol] * right[iLeftCol][iRightCol];
                }
                product[iLeftRow][iRightCol] = sum;
            }
        }
        return product;
    }

    template <typename DataType>
    std::vector<std::vector<DataType>> Matrix<DataType>::getScale(const std::vector<std::vector<DataType>>& data,
                                                                  const DataType&                           factor) const {
        size_t rows = data.size();
        size_t cols = data[0].size();

        // factor is 1 then dont need to do anything
        if (factor == 1)
            return data;
        // factor is 0 then return matrix of zeros
        else if (factor == 0)
            return std::vector<std::vector<DataType>>(rows, std::vector<DataType>(cols, 0));

        std::vector<std::vector<DataType>> scale(rows, std::vector<DataType>(rows, 0));

        for (int iRow = 0; iRow < rows; ++iRow) {
            for (int iCol = 0; iCol < cols; ++iCol) {
                scale[iRow][iCol] = data_[iRow][iCol] * factor;
            }
        }
        return scale;
    }

    template <typename DataType>
    void Matrix<DataType>::appendRow(size_t rowIdx, const std::vector<DataType>& row) {
        assert(row.size() == cols_);
        data_.insert(data_.begin() + rowIdx, row); // basic pointer math (iterators are pointers)
        ++rows_;
    }

    template <typename DataType>
    void Matrix<DataType>::appendColumn(size_t colIdx, const std::vector<DataType>& column) {
        assert(column.size() == rows_);
        for (size_t iRow = 0; iRow < rows_; ++iRow) {
            data_[iRow].insert(data_[iRow].begin() + colIdx, column[iRow]); // basic pointer math (iterators are pointers)
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
        for (size_t iRow = 0; iRow < rows_; ++iRow) {
            data_[iRow].erase(data_[iRow].begin() + colIdx); // basic pointer math (iterators are pointers)
        }
        --cols_;
    }

    template <typename DataType>
    std::vector<DataType> Matrix<DataType>::getEndRow() const {
        return *data_.back(); // .back() - points to last element, .end()- points to one addres past last element (null)
    }

    template <typename DataType>
    std::vector<DataType> Matrix<DataType>::getBeginRow() const {
        return *data_.begin();
    }

    template <typename DataType>
    std::vector<DataType> Matrix<DataType>::getEndColumn() const {
        std::vector<DataType> ret(rows_, 0);
        for (int iRow = 0; iRow < rows_; ++iRow) {
            ret.emplace_back(
                *data_[iRow].back()); // .back() - points to last element, .end()- points to one addres past last element (null)
        }
        return ret;
    }

    template <typename DataType>
    std::vector<DataType> Matrix<DataType>::getBeginColumn() const {
        std::vector<DataType> ret(rows_, 0);
        for (int iRow = 0; iRow < rows_; ++iRow) {
            ret.emplace_back(*data_[iRow].end());
        }
        return ret;
    }

    template <typename DataType>
    inline void Matrix<DataType>::resize(size_t newRows, size_t newCols) {
        assert(newRows > 0);
        assert(newCols > 0);

        if (newRows < rows_ || newCols < cols_)
            LOG("Shrinking size, losing data!");

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
    std::vector<DataType> Matrix<DataType>::getDiag() const {
        if (cols_ != rows_)
            throw std::runtime_error("No diag for non-square matrix!");

        std::vector<DataType> ret(rows_, 0);

        for (int iDiag = 0; iDiag < rows_; ++iDiag) {
            ret[iDiag] = data_[iDiag][iDiag];
        }
        return ret;
    }

    template <typename DataType>
    void Matrix<DataType>::transpose() {
#if USE_EXCEPTIONS
        try {
            data_ = std::move(getTranspose(data_));
        } catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        data_ = std::move(getTranspose(data_));
#endif
        // swap dimensions
        std::swap(rows_, cols_);
    }

    template <typename DataType>
    void Matrix<DataType>::invert() {
#if USE_EXCEPTIONS
        try {
            data_ = std::move(getInverse(data_));
        } catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        data_ = std::move(getInverse(data_));
#endif
    }

    // addition operator (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator+(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(other.rows_ == rows_);
        assert(other.cols_ == rows_);

        Matrix<DataType> result(other.rows_, other.cols_);

        for (int iRow = 0; iRow < rows_; ++iRow) {
            for (int iCol = 0; iCol < cols_; ++iCol) {
                result.data_[iRow][iCol] = data_[iRow][iCol] + other.data_[iRow][iCol];
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

        for (int iRow = 0; iRow < rows_; ++iRow) {
            for (int iCol = 0; iCol < cols_; ++iCol) {
                data_[iRow][iCol] += other.data_[iRow][iCol];
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

        for (int iRow = 0; iRow < rows_; ++iRow) {
            for (int iCol = 0; iCol < cols_; ++iCol) {
                result.data_[iRow][iCol] = data_[iRow][iCol] - other.data_[iRow][iCol];
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

        for (int iRow = 0; iRow < rows_; ++iRow) {
            for (int iCol = 0; iCol < cols_; ++iCol) {
                data_[iRow][iCol] -= other.data_[iRow][iCol];
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

        data_ = std::move(getScale(data_, factor)); // std::vector move operator
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
        data_ = std::move(getScale(data_, 1 / factor)); // std::vector move operator
        return *this;
    }

    // multiplication operator for matrixes (return copy of result)
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator*(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(cols_ == other.rows_);

#if USE_EXCEPTIONS
        try {
            Matrix<DataType> result(std::move(getProduct(data_, other.data_)));
        } // Matrix cstr no.1
        catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        Matrix<DataType> result(std::move(getProduct(data_, other.data_))); // Matrix cstr no.1
#endif
        result.rows_ = rows_;
        result.cols_ = other.cols_;
        return result;
    }

    // multiplication operator for matrixes (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator*=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(cols_ == other.rows_);

#if USE_EXCEPTIONS
        try {
            data_ = std::move(getProduct(data_, other.data_));
        } catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        data_ = std::move(getProduct(data_, other.data_));
#endif
        rows_ = rows_;
        cols_ = other.cols_;
        return *this;
    }

    // division operator for matrixes (return copy of result), using multiplication operator
    template <typename DataType>
    inline Matrix<DataType> Matrix<DataType>::operator/(const Matrix<DataType>& other) const {
        // assert correct dimensions
        assert(cols_ == other.rows_);

#if USE_EXCEPTIONS
        // division is multiplication by inverse
        try {
            Matrix<DataType> result(std::move(getProduct(data_, getInverse(other.data_))));
        } catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        Matrix<DataType> result(std::move(getProduct(data_, getInverse(other.data_))));
#endif
        result.rows_ = rows_;
        result.cols_ = other.cols_;
        return result;
    }

    // division operator for matrixes (return reference to this (lhs)), using multiplication operator
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator/=(const Matrix<DataType>& other) {
        // assert correct dimensions
        assert(cols_ == other.rows_);

#if USE_EXCEPTIONS
        // division is multiplication by inverse
        try {
            data_ = std::move(getProduct(data_, getInverse(other.data_)));
        } // std::vector move operator
        catch (const std::runtime_error& error) {
            LOG(error);
        }
#else
        // division is multiplication by inverse
        data_ = std::move(getProduct(data_,
                                     getInverse(other.data_))); // std::vector move operator
#endif
        rows_ = rows_;
        cols_ = other.cols_;
        return *this;
    }

    // copy operator (return reference to this (lhs))
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator=(const Matrix<DataType>& other) {
        // same container inside objects, no need to move entire object
        if (data_ == other.data_)
            return *this;

        data_ = other.data_;
        cols_ = other.cols_;
        rows_ = other.rows_;
        return *this;
    }

    // move operator (return reference to this (lhs)), noexcept since move cstrs/operators should not throw
    template <typename DataType>
    inline Matrix<DataType>& Matrix<DataType>::operator=(Matrix<DataType>&& other) noexcept {
        // same container inside objects, no need to move entire object
        if (data_ == other.data_)
            return *this;

        // NOTICE how its non const reference, because the object we are moving from is
        // actually the object we are "stealing" from, so it will be left "deinitialized" and shouldnt
        // be used after.  If it had any pointers, we would explicitly need to define our own copy constructors,
        // because default one would just shallow copy them (copy just the pointer), which would result in two copies of pointer to same
        // memory So the copying should copy the memory rather than pointer, and moving should "Steal" the pointer and set the pointer to
        // null in object stolen from. But if you use smart pointers, you dont need to worry about this, as unique_ptr would be simply moved
        // (as there cannot be another copy), and shared_ptr would be copied, as they are designed to control handling many copies of the
        // same ptr (only the destructor of the last shared_ptr will free the memory, they are implemented use static members to count how
        // many resource users there are)

        // move operator of data_ (std::vector) (stealing this memmory from other!)
        data_ = std::move(other.data_);
        // no move operator for trivial
        cols_ = other.cols_;
        rows_ = other.rows_;
        return *this;
    }

    // index operator for rows (return reference of result)
    template <typename DataType>
    inline std::vector<DataType> Matrix<DataType>::operator()(size_t rowIdx) const {
        return data_[rowIdx];
    }

    // index operator for rows/cols (elements) (return reference of result)
    template <typename DataType>
    inline DataType Matrix<DataType>::operator()(size_t rowIdx, size_t colIdx) const {
        return data_[rowIdx][colIdx];
    }

} // namespace mtx

#endif // MATRIX_HPP
