#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "common.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <exception>
#include <fmt/core.h>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector {
    public:
        using Size = std::size_t;
        using VectorData = std::vector<Value>;
        using VectorInitializer = std::initializer_list<Value>;
        using Error = std::runtime_error;

        [[nodiscard]] static constexpr Vector make_vector(VectorInitializer const data)
        {
            Vector vector{};
            vector.data_.reserve(data.size());
            for (auto const elem : data) {
                vector.data_.push_back(elem);
            }
            return vector;
        }

        [[nodiscard]] static constexpr Vector make_zeros(Size const size)
        {
            Vector vector{};
            vector.data_.reserve(size);
            for (Size i{0}; i < size; ++i) {
                vector.data_.emplace_back();
            }
            return vector;
        }

        [[nodiscard]] static constexpr Vector make_ones(Size const size)
        {
            Vector vector{};
            vector.data_.reserve(size);
            for (Size i{0}; i < size; ++i) {
                vector.data_.push_back(Value{1});
            }
            return vector;
        }

        [[nodiscard]] static constexpr Vector sum(Vector const& left, Vector const& right)
        {
            if (left.size() != right.size()) {
                throw Error{"Incorrect dimensions!\n"};
            }

            auto sum{make_zeros(left.size())};
            for (Size i{0}; i < left.size(); ++i) {
                sum[i] = left[i] + right[i];
            }
            return sum;
        }

        [[nodiscard]] static constexpr Vector difference(Vector const& left, Vector const& right)
        {
            if (left.size() != right.size()) {
                throw Error{"Incorrect dimensions!\n"};
            }

            auto difference{make_zeros(left.size())};
            for (Size i{0}; i < left.size(); ++i) {
                difference[i] = left[i] - right[i];
            }
            return difference;
        }

        [[nodiscard]] static constexpr Vector scale(Vector const& vector, Value const factor)
        {
            Size const size{vector.size()};

            if (factor == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf!\n"};
            } else if (factor == 1) {
                return vector;
            } else if (factor == 0) {
                return make_zeros(size);
            }

            auto scale{make_zeros(size)};
            for (Size i{0}; i < size; ++i) {
                scale[i] = vector[i] * factor;
            }
            return scale;
        }

        constexpr Vector() noexcept = default;

        explicit constexpr Vector(VectorInitializer const data) : data_{data}
        {}

        constexpr Vector(Size const size) : data_{make_zeros(size)}
        {}

        explicit constexpr Vector(VectorData&& data) noexcept : data_{std::forward<VectorData>(data)}
        {}

        explicit constexpr Vector(VectorData const& data) : data_{data}
        {}

        constexpr Vector(Vector const& other) = default;

        constexpr Vector(Vector&& other) noexcept = default;

        constexpr ~Vector() noexcept = default;

        constexpr Vector& operator=(Vector const& other) = default;

        constexpr Vector& operator=(Vector&& other) noexcept = default;

        constexpr void operator=(this Vector& self, VectorInitializer const data)
        {
            self.data_ = data;
        }

        constexpr void operator=(this Vector& self, VectorData&& data) noexcept
        {
            self.data_ = std::forward<VectorData>(data);
        }

        constexpr void operator=(this Vector& self, VectorData const& data)
        {
            self.data_ = data;
        }

        constexpr Vector& operator+=(this Vector& self, Vector const& other)
        {
            try {
                self = Vector::sum(self, other);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        constexpr Vector& operator-=(this Vector& self, Vector const& other)
        {
            try {
                self = Vector::difference(self, other);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        constexpr Vector& operator*=(this Vector& self, Value const factor)
        {
            try {
                self = Vector::scale(self, factor);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        constexpr Vector& operator/=(this Vector& self, Value const factor)
        {
            try {
                self = Vector::scale(self, 1 / factor);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        friend constexpr Vector operator+(Vector const& left, Vector const& right)
        {
            try {
                return Vector::sum(left, right);
            } catch (Error const& error) {
                throw error;
            }
        }

        friend constexpr Vector operator-(Vector const& left, Vector const& right)
        {
            try {
                return Vector::difference(left, right);
            } catch (Error const& error) {
                throw error;
            }
        }

        friend constexpr Vector operator*(Value const factor, Vector const& vector)
        {
            try {
                return Vector::scale(vector, factor);
            } catch (Error const& error) {
                throw error;
            }
        }

        friend constexpr Vector operator*(Vector const& vector, Value const factor)
        {
            try {
                return Vector::scale(vector, factor);
            } catch (Error const& error) {
                throw error;
            }
        }

        friend constexpr Vector operator/(Vector const& vector, Value const factor)
        {
            try {
                return Vector::scale(vector, 1 / factor);
            } catch (Error const& error) {
                throw error;
            }
        }

        explicit constexpr operator VectorData(this Vector&& self) noexcept
        {
            return std::forward<Vector>(self).data_;
        }

        explicit constexpr operator VectorData(this Vector const& self) noexcept
        {
            return self.data_;
        }

        [[nodiscard]] constexpr Value& operator[](this Vector& self, Size const elem) noexcept
        {
            if (elem > self.size()) {
                throw Error{"Wrong dimensions\n"};
            }

            return self.data_[elem];
        }

        [[nodiscard]] constexpr Value const& operator[](this Vector const& self, Size const elem) noexcept
        {
            if (elem > self.size()) {
                throw Error{"Wrong dimensions\n"};
            }

            return self.data_[elem];
        }

        [[nodiscard]] constexpr bool operator<=>(this Vector const& self, Vector const& other) noexcept = default;

        constexpr void print(this Vector const& self) noexcept
        {
            fmt::print("[");
            for (auto& elem : self.data_) {
                fmt::print("{}", elem);
                if (elem != self.data_.back()) {
                    fmt::print(", ");
                }
            }
            fmt::print("]\n");
        }

        [[nodiscard]] constexpr VectorData const& data(this Vector const& self) noexcept
        {
            return self.data_;
        }

        [[nodiscard]] constexpr VectorData&& data(this Vector&& self) noexcept
        {
            return std::forward<Vector>(self).data_;
        }

        constexpr void data(this Vector& self, VectorInitializer const data) noexcept
        {
            self.data_ = data;
        }

        constexpr void data(this Vector& self, VectorData&& data) noexcept
        {
            self.data_ = std::forward<VectorData>(data);
        }

        constexpr void data(this Vector& self, VectorData const& data)
        {
            self.data_ = data;
        }

        constexpr void swap(this Vector& self, Vector& other)
        {
            std::swap(self.data_, other.data_);
        }

        constexpr void reserve(this Vector& self, Size const size)
        {
            self.data_.reserve(size);
        }

        constexpr void resize(this Vector& self, Size const size)
        {
            self.data_.resize(size);
        }

        constexpr void erase(this Vector& self)
        {
            self.data_.erase();
        }

        constexpr void clear(this Vector& self)
        {
            self.data_.clear();
        }

        [[nodiscard]] constexpr bool is_empty(this Vector const& self) noexcept
        {
            return self.size() == 0;
        }

        [[nodiscard]] constexpr Size size(this Vector const& self) noexcept
        {
            return self.data_.size();
        }

    private:
        VectorData data_{};
    };

}; // namespace Linalg

#endif // VECTOR_HPP