#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>

#include "config.h"


namespace mtx {


    template <typename T>
    class Matrix {
        public:
            // CUSTOM CONSTRUCTORS
            Matrix(size_t rows = 0, size_t cols = 0, const T &value = 0); // 0
            Matrix(const std::vector<std::vector<T>> &&otherData); // 1
            Matrix(const std::vector<T> &diag); // 2
            
            // ASSINGMENT CONSTRUCTORS AND DESTRUCTOR (RULE OF 5)
            Matrix(const Matrix<T> &other) = default;
            Matrix(Matrix<T> &&other) = default;
            ~Matrix() = default;

            // INTERFACE FUNCTION MEMBERS 
            void transpose();
            void invert();
            inline void appendRow(size_t rowIdx, const std::vector<T> &row);
            inline void appendColumn(size_t colIdx, const std::vector<T> &column);
            inline void deleteRow(size_t rowIdx);
            inline void deleteColumn(size_t colIdx);           
            inline void resize(size_t newRows, size_t newCols);
            inline size_t getRows() const;
            inline size_t getCols() const; 
            inline std::vector<T> getEndRow() const;
            inline std::vector<T> getBeginRow() const;
            inline std::vector<T> getEndColumn() const;
            inline std::vector<T> getBeginColumn() const;
            inline std::vector<T> getDiag() const;

            // MATH OPERATORS
            inline Matrix<T> operator*(const T &factor) const;
            inline Matrix<T> operator/(const T &factor) const;
            inline Matrix<T> operator+(const Matrix<T> &other) const;
            inline Matrix<T> operator-(const Matrix<T> &other) const;
            inline Matrix<T> operator*(const Matrix<T> &other) const;
            inline Matrix<T> operator/(const Matrix<T> &other) const;
            inline Matrix<T> &operator*=(const T &factor);
            inline Matrix<T> &operator/=(const T &factor);
            inline Matrix<T> &operator+=(const Matrix<T> &other);
            inline Matrix<T> &operator-=(const Matrix<T> &other);
            inline Matrix<T> &operator*=(const Matrix<T> &other);
            inline Matrix<T> &operator/=(const Matrix<T> &other);
            inline std::vector<T> &operator()(size_t rowIdx) const;
            inline T &operator()(size_t rowIdx, size_t colIdx) const;

            // CUSTOM ASSINGMENT OPERATORS (NOW NEED TO DEFINE REST FROM RULE OF 5)
            inline Matrix<T> &operator=(const Matrix<T> &other);
            inline Matrix<T> &operator=(Matrix<T> &&other) noexcept;

        private:
            // PRIVATE FUNCTION MEMBERS 
            T getDeterminant(const std::vector<std::vector<T>> &data, size_t matrixDims) const;
            std::vector<std::vector<T>> getMinor(const std::vector<std::vector<T>> &data, size_t rowIdx, size_t colIdx, size_t matrixDims) const;
            std::vector<std::vector<T>> getTranspose(const std::vector<std::vector<T>> &data) const;
            std::vector<std::vector<T>> getAdjoint(const std::vector<std::vector<T>> &data) const;
            std::vector<std::vector<T>> getInverse(const std::vector<std::vector<T>> &data) const;
            std::vector<std::vector<T>> getLowerTriangular(const std::vector<std::vector<T>> &data) const;
            std::vector<std::vector<T>> getUpperTriangular(const std::vector<std::vector<T>> &data) const;
            std::vector<std::vector<T>> getScale(const std::vector<std::vector<T>> &data, const T &factor) const;
            std::vector<std::vector<T>> getProduct(const std::vector<std::vector<T>> &left, const std::vector<std::vector<T>> &right) const;

            // DATA MEMBERS
            size_t rows_ {}; // first dim
            size_t cols_ {}; // second dim
            T det_ {};
            std::vector<std::vector<T>> data_ {}; // vector of vectors
    };



    template <typename T>
    Matrix<T>::Matrix(size_t rows, size_t cols, const T &value) :   rows_(rows), 
                                                                    cols_(cols), 
                                                                    data_(rows_, std::vector<T>(cols_, value)) { // this cstr takes both rvalue and lvalue, no need for std::forward of second argument
        // assert that either both dimensions were specified, or neither
        assert((rows == 0 && cols == 0) || (rows != 0 && cols != 0));

        // for (int iRow = 0; iRow < rows_; ++iRow) {
        //     data_.emplace_back(std::vector<T>); // emplace_back taking rvalue
        //     for (int iCol = 0; iCol < cols_; ++iCol) {
        //         // data_[iRow].emplace_back(value);
        //         data_.end()->emplace_back(value); // emplace_back taking rvalue
        //     }
        // }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T> &diag) : rows_(diag.size()),
                                                    cols_(rows_),
                                                    data_(rows_, std::vector<T>(cols_, 0)) {  // this cstr takes both rvalue and lvalue, no need for std::forward of second argument
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
    template <typename T>                                                 // (othetData is passed by rvalue ref, so its move cstr)
    Matrix<T>::Matrix(const std::vector<std::vector<T>> &&otherData) : data_(otherData),
                                                                        rows_(data_.size()),
                                                                        cols_(data_[0].size()) {
    } 

    // if you dont need a custom destructor, then don't, as creating your own destructors (aswell as copy constructors) can disable the compilers
    // ability to optimize assembly instructions using move operations 
    // template <typename T>     
    // Matrix<T>::~Matrix() {
    //     // compiler will use default (most optimized) self created destructor, std::vector will take care of its memory on the heap
    // }

    template <typename T>     
    std::vector<std::vector<T>> Matrix<T>::getMinor(const std::vector<std::vector<T>> &data, size_t rowIdx, size_t colIdx, size_t matrixDims) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        #if USE_EXCEPTIONS
        if (rows != cols) throw std::runtime_error("No cofactor for non-square matrix!");
        #endif

        // assert cofactor isnt calculated for minor bigger than data 
        assert(matrixDims < rowIdx && matrixDims < colIdx);
        #if USE_EXCEPTIONS
        if (matrixDims >= rowIdx || matrixDims >= colIdx) throw std::runtime_error("Wrong minor!");
        #endif

        // minor is scalar, can omit later code
        if (matrixDims == 0) return data;
        
        // result object
        std::vector<std::vector<T>> minor(matrixDims, std::vector<T>(matrixDims, 0));

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

    template <typename T>
    T Matrix<T>::getDeterminant(const std::vector<std::vector<T>> &data, size_t matrixDims) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        #if USE_EXCEPTIONS
        if (rows != cols) throw std::runtime_error("No cofactor for non-square matrix!");
        #endif

        // assert minor isnt bigger than data  
        assert(matrixDims < rows && matrixDims < cols);
        #if USE_EXCEPTIONS
        if (matrixDims >= rows || matrixDims >= cols) throw std::runtime_error("Wrong data!");
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
        T det = 0;

        // sign multiplier
        T sign = 1;
        std::vector<std::vector<T>> minor(matrixDims, std::vector<T>(cols, matrixDims)); // this cstr takes both rvalue and lvalue, no need for std::forward of second argument
        // iterate for each element of first row
        for (size_t iCol = 0; iCol < matrixDims; ++iCol) {
            // cofactor of data[0][iCol]
            #if USE_EXCEPTIONS
            try {minor = getMinor(data, 0, iCol, matrixDims);} // std::vector move operator
            catch (const std::runtime_error &error) {LOG(error);}

            try {det += sign * data[0][iCol] * determinant(minor, matrixDims - 1);}
            catch (const std::runtime_error &error) {LOG(error);}
            #else 
                minor = getMinor(data, 0, iCol, matrixDims); // std::vector move operator
                det += sign * data[0][iCol] * getDeterminant(minor, matrixDims - 1);
            #endif
            
            // alternate sign
            sign *= -1;
        }

        return det;
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getTranspose(const std::vector<std::vector<T>> &data) const {
        size_t newRows = data[0].size();
        size_t newCols = data.size();

        // data is scalar, can omit later code
        if (newRows == newCols == 1) return data;

        // result object
        std::vector<std::vector<T>> transpose(newRows, std::vector<T>(newCols, 0)); // this cstr takes both rvalue and lvalue, no need for std::forward of second argument
        
        for (int iRow = 0; iRow < newRows; ++iRow) {
            for (int iCol = 0; iCol < newCols; ++iCol) {
                transpose[iRow][iCol] = data[iCol][iRow];
            }
        }
        return transpose;
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getAdjoint(const std::vector<std::vector<T>> &data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        // throw exception if non-square
        #if USE_EXCEPTIONS
        if (rows != cols) throw std::runtime_error("No adjoint for non-square matrix!");
        #endif

        // matrix square 
        size_t matrixDims = rows = cols;
        // data is scalar, can omit later code
        if (matrixDims == 1) return data;

        std::vector<std::vector<T> > complement(matrixDims, std::vector<T>(matrixDims, 0));

        // sign multiplier
        T sign = 1;
        std::vector<std::vector<T> > minor(matrixDims, std::vector<T>(matrixDims, 0));
        for (int iRow = 0; iRow < matrixDims; ++iRow) {
            for (int iCol = 0; iCol < matrixDims; iCol++) {
                // get cofactor of data[iRow][iCol]
                #if USE_EXCEPTIONS
                try {minor = getMinor(data, iRow, iCol, matrixDims);}
                catch (const std::runtime_error &error) {LOG(error);}
                // sign of adj[iCol][iRow] positive if sum of row and column indexes is even
                if ((iRow + iCol) % 2 == 0) sign = 1;
                else sign = -1;

                // complement is matrix of determinants of minors with alternating signs!!!
                try {complement[iRow][iCol] = (sign) * (determinant(minor, matrixDims - 1));}
                catch (const std::runtime_error &error) {LOG(error);}
                #else 
                minor = getMinor(data, iRow, iCol, matrixDims);
                // sign of adj[iCol][iRow] positive if sum of row and column indexes is even
                if ((iRow + iCol) % 2 == 0) sign = 1;
                else sign = -1;

                // complement is matrix of determinants of minors with alternating signs!!!
                complement[iRow][iCol] = (sign) * (getDeterminant(minor, matrixDims - 1));
                #endif
            }
        }
        // result object
        // adjoint is transposed of complement matrix
        return std::vector<std::vector<T>>(getTranspose(complement)); // dont declare if all you need is to return or copy it into collection, instead just use constructor without declaring varaible if it isnt used
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getInverse(const std::vector<std::vector<T>> &data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        /// throw exception if non-square
        #if USE_EXCEPTIONS
        if (rows != cols) throw std::runtime_error("No inverse for non-square matrix!");
        #endif

        // matrix square 
        size_t matrixDims = rows = cols;
        // data is scalar, can omit later code
        if (matrixDims == 1) return data;

        T det = getDeterminant(data, matrixDims);
        // assert correct determinant 
        assert(det != 0);
        // throw exception if singular
        #if USE_EXCEPTIONS
        if (det == 0) throw std::runtime_error("No inverse for singular matrix!");
        #endif
        
        
        #if USE_EXCEPTIONS 
        try {std::vector<std::vector<T>> adjoint(getAdjoint(data));}
        catch (const std::runtime_error &error) {LOG(error);}
        #else 
        std::vector<std::vector<T>> adjoint(getAdjoint(data));
        #endif

        // inverse is adjoint matrix divided by det factor
        // division is multiplication by inverse
        return std::vector<std::vector<T>>(getScale(adjoint, 1/det)); // Matrix cstr no.1;
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getUpperTriangular(const std::vector<std::vector<T>> &data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        // throw exception if non-square
        #if USE_EXCEPTIONS
        if (cols != rows) throw std::runtime_error("No upper-triangular for non-square matrix!");
        #endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1) return data;

        // result object
        // upper triangular is just transpose of lower triangular (cholesky- A = L*L^T)
        return std::vector<std::vector<T>> (getTranspose(std::forward(getLowerTriangular(data)))); // std::forward as getTranspose needs lvalue, but leaving rvalue in std::vector cstr to call move cstr!
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getLowerTriangular(const std::vector<std::vector<T>> &data) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        // assert correct dimensions 
        assert(rows == cols);
        // throw exception if non-square
        #if USE_EXCEPTIONS
        if (cols != rows) throw std::runtime_error("No lower-triangular for non-square matrix!");
        #endif

        // matrix square
        size_t matrixDims = rows = cols;
        // data is scalar
        if (matrixDims == 1) return data;

        // result object
        std::vector<std::vector<T>> lowerTriangular(matrixDims, std::vector<T>(matrixDims, 0)); // this cstr takes both rvalue and lvalue, no need for std::forward of second argument

        // decomposing data matrix into lower triangular
        for (int iRow = 0; iRow < matrixDims; ++iRow) {
            for (int iCol = 0; iCol <= iRow; ++iCol) {
                T sum = 0;

                // summation for diagonals
                if (iCol == iRow) 
                {
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

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getProduct(const std::vector<std::vector<T>> &left, 
                                                            const std::vector<std::vector<T>> &right) const {

        size_t leftRows = left.size(), rightRows = right.size();
        size_t leftCols = left[0].size(), rightCols = right[0].size();

        // assert correct dimensions 
        assert(leftCols == rightCols);
        // throw exception to incorrect dimensions
        #if USE_EXCEPTIONS
        if (leftCols != rightCols) throw std::runtime_error("No cholseky for non-square matrix!");
        #endif

        size_t productRows = leftRows;
        size_t productCols = rightCols;
        std::vector<std::vector<T>> product(productRows, std::vector<T>(productCols, 0)); // this cstr takes both rvalue and lvalue, no need for std::forward of second argument

        for (size_t iLeftRow = 0; iLeftRow < leftRows; ++iLeftRow) {
            for (size_t iRightCol = 0; iRightCol < rightCols; ++iRightCol) {
                T sum = 0;
                for (size_t iLeftCol = 0; iLeftCol < leftCols; ++iLeftCol) {
                    sum += left[iLeftRow][iLeftCol] * right[iLeftCol][iRightCol];
                }
                product[iLeftRow][iRightCol] = sum;
            }
        }
        return product;
    }

    template <typename T>
    std::vector<std::vector<T>> Matrix<T>::getScale(const std::vector<std::vector<T>> &data, const T &factor) const {
        size_t rows = data.size();
        size_t cols = data[0].size();
        
        // factor is 1 then dont need to do anything
        if (factor == 1) return data;
        // factor is 0 then return matrix of zeros
        else if (factor == 0) return std::vector<std::vector<T>>(rows, std::vector<T>(cols, 0));

        std::vector<std::vector<T>> scale(rows, std::vector<T>(rows, 0));

        for (int iRow = 0; iRow < rows; ++iRow) {
            for (int iCol = 0; iCol < cols; ++iCol) {
                scale[iRow][iCol] = data_[iRow][iCol] * factor;
            }   
        }
        return scale;
    }



    template <typename T>
    void Matrix<T>::appendRow(size_t rowIdx, const std::vector<T> &row) {
        assert(row.size() == cols_);
        data_.insert(data_.begin() + rowIdx, row); // basic pointer math (iterators are pointers)
        ++rows_;
    }

    template <typename T>
    void Matrix<T>::appendColumn(size_t colIdx, const std::vector<T> &column) {
        assert(column.size() == rows_);
        for (size_t iRow = 0; iRow < rows_; ++iRow) {
            data_[iRow].insert(data_[iRow].begin() + colIdx, column[iRow]); // basic pointer math (iterators are pointers)
        }
        ++cols_;
    }

    template <typename T>
    void Matrix<T>::deleteRow(size_t rowIdx) {
        assert(rowIdx <= rows_);
        data_.erase(data_.begin() + rowIdx); // basic pointer math (iterators are pointers)
        --rows_;
    }

    template <typename T>
    void Matrix<T>::deleteColumn(size_t colIdx) {
        assert(colIdx <= cols_);
        for (size_t iRow = 0; iRow < rows_; ++iRow) {
            data_[iRow].erase(data_[iRow].begin() + colIdx); // basic pointer math (iterators are pointers)
        }
        --cols_;
    }

    template <typename T>
    std::vector<T> Matrix<T>::getEndRow() const {
        return *data_.back(); // .back() - points to last element, .end()- points to one addres past last element (null)
    }

    template <typename T>
    std::vector<T> Matrix<T>::getBeginRow() const {
        return *data_.begin();
    }

    template <typename T>
    std::vector<T> Matrix<T>::getEndColumn() const {
        std::vector<T> ret(rows_, 0);
        for (int iRow = 0; iRow < rows_; ++iRow) {
            ret.emplace_back(*data_[iRow].back()); // .back() - points to last element, .end()- points to one addres past last element (null)
        }
        return ret;
    }

    template <typename T>
    std::vector<T> Matrix<T>::getBeginColumn() const {
        std::vector<T> ret(rows_, 0);
        for (int iRow = 0; iRow < rows_; ++iRow) {
            ret.emplace_back(*data_[iRow].end());
        }
        return ret;
    }

    template <typename T>
    inline void Matrix<T>::resize(size_t newRows, size_t newCols) {
        assert(newRows > 0);
        assert(newCols > 0);

        if (newRows < rows_ || newCols < cols_) LOG("Shrinking size, losing data!");

        rows_ = newRows;
        cols_ = newCols;

        data_.resize(rows_);
        for (auto &row : data_) {
            row.resize(cols_);
        }
    }

    template <typename T>
    inline size_t Matrix<T>::getRows() const {
        return rows_;
    }

    template <typename T>
    inline size_t Matrix<T>::getCols() const {
        return cols_;
    }

    template <typename T>    
    std::vector<T> Matrix<T>::getDiag() const {
        if (cols_ != rows_) throw std::runtime_error("No diag for non-square matrix!");

        std::vector<T> ret(rows_, 0);

        for (int iDiag = 0; iDiag < rows_; ++iDiag) {
                ret[iDiag] = data_[iDiag][iDiag];
        }
        return ret;
    }




    template <typename T>
    void Matrix<T>::transpose() {
        #if USE_EXCEPTIONS
        try {data_ = getTranspose(data_);} // function already returning rvalue, no need to use move operator
        catch (const std::runtime_error& error) {LOG(error);}
        #else 
        data_ = getTranspose(data_); // function already returning rvalue, no need to use move operator
        #endif
        // swap dimensions
        std::swap(rows_, cols_); 
    }

    template <typename T>
    void Matrix<T>::invert() {
        #if USE_EXCEPTIONS
        try {data_ = getInverse(data_);} // function already returning rvalue, no need to use move operator 
        catch (const std::runtime_error& error) {LOG(error);}
        #else 
        data_ = getInverse(data_);
        #endif
    }

    // template <typename T>
    // void Matrix<T>::cholesky() {
    //     #if USE_EXCEPTIONS
    //     try {data_ = getCholesky(data_);} // function already returning rvalue, no need to use move operator 
    //     catch (const std::runtime_error& error) {LOG(error);}
    //     #else 
    //     data_ = getCholesky(data_);
    //     #endif
    // }



    // addition operator (return copy of result)
    template <typename T>
    inline Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
        // assert correct dimensions 
        assert(other.rows_ == this->rows_);
        assert(other.cols_ == this->rows_);

        Matrix<T> result(other.rows_, other.cols_);

        for (int iRow = 0; iRow < this->rows_; ++iRow) {
            for (int iCol = 0; iCol < this->cols_; ++iCol) {
                result.data_[iRow][iCol] = this->data_[iRow][iCol] + other.data_[iRow][iCol];
            }   
        }
        return result;
    }

    // addition operator (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other) {
        // assert correct dimensions 
        assert(other.rows_ == this->rows_);
        assert(other.cols_ == this->rows_);

        for (int iRow = 0; iRow < this->rows_; ++iRow) {
            for (int iCol = 0; iCol < this->cols_; ++iCol) {
                this->data_[iRow][iCol] += other.data_[iRow][iCol];
            }   
        }
        return *this;
    }

    // substraction operator (return copy of result)
    template <typename T>
    inline Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
        // assert correct dimensions 
        assert(other.rows_ == this->rows_);
        assert(other.cols_ == this->rows_);

        Matrix<T> result(other.rows_, other.cols_);

        for (int iRow = 0; iRow < this->rows_; ++iRow) {
            for (int iCol = 0; iCol < this->cols_; ++iCol) {
                result.data_[iRow][iCol] = this->data_[iRow][iCol] - other.data_[iRow][iCol];
            }   
        }
        return result;
    }

    // substraction operator (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &other) {
        // assert correct dimensions 
        assert(other.rows_ == this->rows_);
        assert(other.cols_ == this->rows_);

        for (int iRow = 0; iRow < this->rows_; ++iRow) {
            for (int iCol = 0; iCol < this->cols_; ++iCol) {
                this->data_[iRow][iCol] -= other.data_[iRow][iCol];
            }   
        }
        return *this;
    }

    // multiplication operator for scalars (return copy of result)
    template <typename T>
    inline Matrix<T> Matrix<T>::operator*(const T &factor) const {
        // factor is 1 then dont need to do anything
        if (factor == 1) return *this;
        // factor is 0 then return matrix of zeros
        else if (factor == 0) return Matrix<T>(rows_, cols_, 0);

        return Matrix<T>(getScale(this->data_, factor)); // Matrix cstr no.1
    }

    // multiplication operator for scalars (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator*=(const T &factor) {
        // factor is 1 then dont need to do anything
        if (factor == 1) return *this;
        // factor is 0 then return zero-ed this
        else if (factor == 0) {
            *this = Matrix<T>(rows_, cols_, 0); // Matrix move operator (*this is already constructed)
            return *this;
        }

        this->data_ = getScale(this->data_, factor); // std::vector move operator
        return *this;
    }

    // division operator for scalars (return copy of result)
    template <typename T>
    inline Matrix<T> Matrix<T>::operator/(const T &factor) const {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1) return *this;
        
        // division is multiplication by inverse
        return Matrix<T>(getScale(this->data_, 1/factor)); // Matrix cstr no.1
    }

    // division operator for scalars (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator/=(const T &factor) {
        // assert no division by 0!!!
        assert(factor != 0);

        // factor is 1 then dont need to do anything
        if (factor == 1) return *this;
        
        // division is multiplication by inverse
        this->data_ = getScale(this->data_, 1/factor); // std::vector move operator
        return *this;
    }


    // multiplication operator for matrixes (return copy of result)
    template <typename T>
    inline Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
        // assert correct dimensions 
        assert(this->cols_ == other.rows_);

        #if USE_EXCEPTIONS
        try {Matrix<T> result(getProduct(this->data_, other.data_));} // Matrix cstr no.1
        catch (const std::runtime_error &error) {LOG(error);}
        #else
        Matrix<T> result(getProduct(this->data_, other.data_)); // Matrix cstr no.1
        #endif
        result.rows_ = this->rows_;
        result.cols_ = other.cols_;
        return result;
    }

    // multiplication operator for matrixes (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other) {
        // assert correct dimensions 
        assert(this->cols_ == other.rows_);

        #if USE_EXCEPTIONS
        try {this->data_ = getProduct(this->data_, other.data_);} // std::vector move operator
        catch (const std::runtime_error &error) {LOG(error);}
        #else
        this->data_ = getProduct(this->data_, other.data_); // std::vector move operator
        #endif
        this->rows_ = this->rows_; 
        this->cols_ = other.cols_;
        return *this;
    }

    // division operator for matrixes (return copy of result), using multiplication operator
    template <typename T>
    inline Matrix<T> Matrix<T>::operator/(const Matrix<T> &other) const {
        // assert correct dimensions 
        assert(this->cols_ == other.rows_);

        #if USE_EXCEPTIONS
        // division is multiplication by inverse
        try {Matrix<T> result(getProduct(this->data_, std::forward(getInverse(other.data_))));} // Matrix cstr no.1, getProduct takes lvalue so std::forward return of getInverse!
        catch (const std::runtime_error &error) {LOG(error);}
        #else
        Matrix<T> result(getProduct(this->data_, std::forward(getInverse(other.data_)))); // Matrix cstr no.1, getProduct takes lvalue so std::forward return of getInverse!
        #endif
        result.rows_ = this->rows_;
        result.cols_ = other.cols_;
        return result;
    }

    // division operator for matrixes (return reference to this (lhs)), using multiplication operator
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator/=(const Matrix<T> &other) {
        // assert correct dimensions 
        assert(this->cols_ == other.rows_);

        #if USE_EXCEPTIONS
        // division is multiplication by inverse
        try {this->data_ = getProduct(this->data_, std::forward(getInverse(other.data_)));} // std::vector move operator, getProduct takes lvalue so std::forward return of getInverse!
        catch (const std::runtime_error &error) {LOG(error);}
        #else
        // division is multiplication by inverse
        this->data_ = getProduct(this->data_, std::forward(getInverse(other.data_))); // std::vector move operator, getProduct takes lvalue so std::forward return of getInverse!
        #endif
        this->rows_ = this->rows_;
        this->cols_ = other.cols_;
        return *this;
    }

    // copy operator (return reference to this (lhs))
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
        // same container inside objects, no need to move entire object 
        if (this->data_  == other.data_) return *this;

        this->data_ = other.data_;
        this->cols_ = other.cols_;
        this->rows_ = other.rows_;
        return *this;
    }

    // move operator (return reference to this (lhs)), noexcept since move cstrs/operators should not throw
    template <typename T>
    inline Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept {
        // same container inside objects, no need to move entire object 
        if (this->data_  == other.data_) return *this;

        // NOTICE how its non const reference, because the object we are moving from is
        // actually the object we are "stealing" from, so it will be left "deinitialized" and shouldnt
        // be used after.  If it had any pointers, we would explicitly need to define our own copy constructors,
        // because default one would just shallow copy them (copy just the pointer), which would result in two copies of pointer to same memory
        // So the copying should copy the memory rather than pointer, and moving should "Steal" the pointer
        // and set the pointer to null in object stolen from. But if you use smart pointers, you dont need to worry about
        // this, as unique_ptr would be simply moved (as there cannot be another copy), and shared_ptr would be copied, as they are
        // designed to control handling many copies of the same ptr (only the destructor of the last shared_ptr will free the memory,
        // they are implemented use static members to count how many resource users there are)  

        // move operator of data_ (std::vector) (stealing this memmory from other!)
        this->data_ = std::move(other.data_);
        // no move operator for trivial
        this->cols_ = other.cols_;
        this->rows_ = other.rows_;
        return *this;
    }

    // index operator for rows (return reference of result)
    template <typename T>
    inline std::vector<T> &Matrix<T>::operator()(size_t rowIdx) const {
        return data_[rowIdx];
    }

    // index operator for rows/cols (elements) (return reference of result)
    template <typename T>
    inline T &Matrix<T>::operator()(size_t rowIdx, size_t colIdx) const {
        return data_[rowIdx][colIdx];
    }


} // namespace mtx


#endif // MATRIX_HPP



/* although you can return ref/ptr to local static variable, you should avoid it (thread safety violated with references to static variables, in RTOS
dont return set pointers/references to static local varaibles, rather copy them!!!)

std::move - makes rvalue from lvalue, to avoid copying when passing to function or using copy operator/constructor (using std::move it passes rvalue), redundant when assigning variable with function return,
as functions already return rvalues (temporary return values)! Moving data is not only faster than copying it, but can be used to explicitly transfer ownership of some memory
from one object to another (unique_ptrs), so that you dont have copies of the pointer to memory, because if you did, you could free it with one copy and then other would be dangling!
std::forward- makes lvalue from rvalue, to avoid copying when passing to function what other function has returned (it returned rvalue), redundant if function we are 'forwarding' takes const ref, as
const ref can take both ref to rvalue and lvalue

*,/,+,-,etc operators return result of this operation, so a copy, without modyfying lhs operator (cannot return reference as returning local result variable)
*=,/=,+=,-=, etc operators return reference to lhs operator, so not a copy (perform operation directly on lhs!)

constexpr- constant expression that can be evaulated at compile time and used to initialize constants, so that you are able
to use these constants (results of constexpr) to allocate parameter specified size of memory or to specify template. Here im using constexpr getters
to static array size (std::array) aswell as constexpr constructors to create them statically with the passed value of constexpr (or you could just call
constexpr locally in a function to initialize the constant/ array size, but I wanted to make this with constructors)
constexpr'essions can be used to evaluate some value at compile time, but be aware, that constexpr variable initialized with constexpr function lives on the stack,
just like any other varaible and all the rules related to variable going out of scope are the same, so do not point to constexpr varaibles that are not 'static'!!!

stack frame- each {} is a stack frame, so an if/while/loop/function/etc is a stack frame, during run time when execution
finishes the code in a stack frame, its cleared (variables go out of scope and their destructors are called). You cannot use variable declared
statically inside if/loop/switch outside of it for the same reason you cant use one from a function, without copying it
to avoid undefined behaviod, to return a result calculated in a loop or if statement, you should copy it first, so make ret variable at beginning of a function

Always remember to copy local stack variables! 

templates- essentially the compiler is putting the constexpr template value put in <> brackets with declaration of object in places of definition of this objects class,
meaning you can have a template for type, value, etc. Very similar to constexpr for values, returning the same things good for having parametrized and elegant, without macros 
(although still not flexible, as everything during compile time) way of allocating given amount of static memory, just need to call constexpr and initialize
const variable with it)!!!

for example doing this is esentially using N() constexpr'ession to calculate value of N at compile time and then using it for the value of template,
which will then exchange all <N> in class code (for this object) for the value of N, which can be used to determine the buffer size at compile time using the
N parameter; its much better than using macro!
const int N = N();
Class<N> obj;

Since C++17, copy intitializaion for POD (plain old data, aggregate types from C) (T t = value) is optimized away to be the same as dircet initialization 
(T t(value) or T t{value}) so for POD types (built-in types, aggregate structs, arrays, pointers) you can just use copy initialization (=), since it wont copy
For non-PODs (non-aggregate, objects, containers, etc) use direct initialization (T t(value) or T t{value}, where value is value passed to constructor of T type!!!)
But theres some caveat, the copy initializaion (T t = x;) allows to narrowing by assigning a value to the variable that it cannot hold, example value bigger than 255 to 8bit type.
The list initialization ( T t{x}; ) will cause error, which can allow you to catch a bug like that quickly. 
So to sum up, use direct initialiation (either () or {}), depending on the type of initialization. () is better for calling constructors, expecially for
object with initializer lists defined, like std containers: std::vector<T> vec(1,2) is vector of 1 T type of value 2 (cstr taking 2 args called), however std::vector<T> vec{1, 2}
is vector of 2 T types, one of value 1, other of value 2 (default cstr + initializer list called because of brackets);
Use () for constructors, use {} for calling initializer lists and default constructors (since it explicitly tells that objects data is
uninitialized, while using () as way of calling default constructor might result in that constructor actually doing some work 
(it just doesnt take any params)), you might not know the definition of that cstr! Use = for PODs, since its clearer, lol

*/
