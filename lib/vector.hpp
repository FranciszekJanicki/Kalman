#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "arithmetic.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <expected>
#include <fmt/core.h>
#include <initializer_list>
#include <utility>
#include <vector>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector {
    public:
        enum struct Error {
            WRONG_DIMS,
        };

        using Size = std::size_t;
        using VectorData = std::vector<Value>;
        using ExpectedVector = std::expected<Vector, Error>;
        using Unexpected = std::unexpected<Error>;

        [[nodiscard]] static constexpr Vector make_vector(std::initializer_list<Value> const data)
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

        [[nodiscard]] static constexpr Vector sum(Vector const& left, Vector const& right) noexcept
        {
            assert(left.size() == right.size());

            auto sum{Vector::make_zeros(left.size())};
            for (Size i{0}; i < left.size(); ++i) {
                sum[i] = left[i] + right[i];
            }
            return sum;
        }

        [[nodiscard]] static constexpr Vector difference(Vector const& left, Vector const& right) noexcept
        {
            assert(left.size() == right.size());

            auto difference{Vector::make_zeros(left.size())};
            for (Size i{0}; i < left.size(); ++i) {
                difference[i] = left[i] - right[i];
            }

            return difference;
        }

        [[nodiscard]] static constexpr Vector scale(Vector const& vector, Value const factor)
        {
            Size const size{vector.size()};

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return vector;
            }
            // factor is 0 then return vectorof zeros
            else if (factor == 0) {
                return Vector::make_zeros(size);
            }

            auto scale{Vector::make_zeros(size)};
            for (Size i{0}; i < size; ++i) {
                scale[i] = vector[i] * factor;
            }
            return scale;
        }

        constexpr Vector() noexcept = default;

        explicit constexpr Vector(std::initializer_list<std::initializer_list<Value> const> const data) : data_{data}
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

        constexpr void operator=(this Vector& self, VectorData&& data) noexcept
        {
            self.data_ = std::forward<VectorData>(data);
        }

        constexpr void operator=(this Vector& self, VectorData const& data)
        {
            self.data_ = data;
        }

        constexpr Vector& operator+=(this Vector& self, Vector const& other) noexcept
        {
            // assert correct dimensions
            assert(other.size() == self.size());

            for (Size i{0}; i < self.size(); ++i) {
                self[i] += other[i];
            }

            return self;
        }

        constexpr Vector& operator-=(this Vector& self, Vector const& other) noexcept
        {
            // assert correct dimensions
            assert(other.size() == self.size());

            for (Size i{0}; i < self.size(); ++i) {
                self[i] -= other[i];
            }

            return self;
        }

        constexpr Vector& operator*=(this Vector& self, Value const& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return self;
            }

            self = scale(self, factor);
            return self;
        }

        constexpr Vector& operator/=(this Vector& self, Value const& factor)
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

        friend constexpr Vector operator+(Vector const& left, Vector const& right)
        {
            // assert correct dimensions
            assert(left.size() == right.size());

            return Vector::sum(left, right);
        }

        friend constexpr Vector operator-(Vector const& left, Vector const& right)
        {
            // assert correct dimensions
            assert(left.size() == right.size());

            return Vector::difference(left, right);
        }

        friend constexpr Vector operator*(Value const& factor, Vector const& vector)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return vector;
            }

            return Vector::scale(vector, factor);
        }

        friend constexpr Vector operator*(Vector const& vector, Value const& factor)
        {
            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return vector;
            }

            return Vector::scale(vector, factor);
        }

        friend constexpr Vector operator/(Vector const& vector, Value const& factor)
        {
            // assert no division by 0!!!
            assert(factor != 0);

            // factor is 1 then dont need to do anything
            if (factor == 1) {
                return vector;
            }

            // division is multiplication by inverse
            return Vector::scale(vector, 1 / factor);
        }

        explicit constexpr operator VectorData(this Vector&& self) noexcept
        {
            return std::forward<Vector>(self).data_;
        }

        explicit constexpr operator VectorData(this Vector const& self) noexcept
        {
            return self.data_;
        }

        [[nodiscard]] constexpr Value& operator[](this Vector& self, Size const elem, Size const column) noexcept
        {
            assert(elem <= self.size());
            return self.data_[elem];
        }

        [[nodiscard]] constexpr Value const&
        operator[](this Vector const& self, Size const elem, Size const column) noexcept
        {
            assert(elem <= self.size());
            return self.data_[elem];
        }

        [[nodiscard]] constexpr bool operator<=>(this Vector const& self, Vector const& other) noexcept = default;

        constexpr void print(this Vector const& self) noexcept
        {
            fmt::print("[");
            for (auto& elem : self.data_) {
                if constexpr (std::is_integral_v<Value>) {
                    fmt::print("%d", static_cast<int>(elem));
                } else if constexpr (std::is_floating_point_v<Value>) {
                    fmt::print("%f", static_cast<float>(elem));
                }
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
        static constexpr const char* error_to_string(Error const error) noexcept
        {
            switch (error) {
                case Error::WRONG_DIMS:
                    return "Wrong dims";
                default:
                    return "None";
            }
        }

        static constexpr void print(Error const error) noexcept
        {
            fmt::print("%s", error_to_string(error));
        }

        VectorData data_{};
    };
}; // namespace Linalg

#endif // VECTOR_HPP