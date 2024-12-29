#ifndef QUATERNION3D_HPP
#define QUATERNION3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>

using Error = std::runtime_error;

namespace Linalg {

    template <Arithmetic Value>
    struct Quaternion3D {
        [[nodiscard]] constexpr Quaternion3D conjugated(this Quaternion3D const& self) noexcept
        {
            return Quaternion3D{self.w, -self.x, -self.y, -self.z};
        }

        constexpr void conjugate(this Quaternion3D& self) noexcept
        {
            self.x = -self.x;
            self.y = -self.y;
            self.z = -self.z;
        }

        [[nodiscard]] constexpr auto magnitude(this Quaternion3D const& self) noexcept
        {
            return std::sqrt(std::pow(self.w, 2) + std::pow(self.x, 2) + std::pow(self.y, 2) + std::pow(self.z, 2));
        }

        [[nodiscard]] constexpr Quaternion3D normalized(this Quaternion3D const& self) noexcept
        {
            const auto im{static_cast<Value>(1) / self.magnitude()};
            return Quaternion3D{self.w * im, self.x * im, self.y * im, self.z * im};
        }

        constexpr void normalize(this Quaternion3D& self) noexcept
        {
            const auto im{static_cast<Value>(1) / self.magnitude()};
            self *= im;
        }

        constexpr Quaternion3D& operator+=(this Quaternion3D& self, Quaternion3D const& other)
        {
            self.w += other.w;
            self.x += other.x;
            self.y += other.y;
            self.z += other.z;
            return self;
        }

        constexpr Quaternion3D& operator-=(this Quaternion3D& self, Quaternion3D const& other)
        {
            try {
                self += (-1 * other);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        constexpr Quaternion3D& operator*=(this Quaternion3D& self, Quaternion3D const& other)
        {
            self.w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z;
            self.x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y;
            self.y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x;
            self.z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w;
            return self;
        }

        constexpr Quaternion3D& operator*=(this Quaternion3D& self, Value const factor)
        {
            if (factor == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf\n"};
            }
            self.w *= factor;
            self.x *= factor;
            self.y *= factor;
            self.z *= factor;
            return self;
        }

        constexpr Quaternion3D& operator/=(this Quaternion3D& self, Value const factor)
        {
            try {
                self *= (1 / factor);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        template <Arithmetic Converted>
        [[nodiscard]] explicit constexpr operator Quaternion3D<Converted>(this Quaternion3D const& self) noexcept
        {
            return Quaternion3D<Converted>{static_cast<Converted>(self.w),
                                           static_cast<Converted>(self.x),
                                           static_cast<Converted>(self.y),
                                           static_cast<Converted>(self.z)};
        }

        [[nodiscard]] constexpr bool operator<=>(this Quaternion3D const& self,
                                                 Quaternion3D const& other) noexcept = default;

        Value w{};
        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(Quaternion3D<Value> const& left, Quaternion3D<Value> const& right) noexcept
    {
        return Quaternion3D<Value>{left.w + right.w, left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator-(Quaternion3D<Value> const& left, Quaternion3D<Value> const& right) noexcept
    {
        return Quaternion3D<Value>{left.w - right.w, left.x - right.x, left.y - right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator*(Quaternion3D<Value> const& left, Quaternion3D<Value> const& right) noexcept
    {
        return Quaternion3D<Value>{left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
                                   left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
                                   left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
                                   left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w};
    }

    template <Arithmetic Value>
    constexpr auto operator*(Quaternion3D<Value> const& quaternion, Value const factor)
    {
        if (factor == std::numeric_limits<Value>::max()) {
            throw Error{"Multiplication by inf\n"};
        }
        return Quaternion3D<Value>{quaternion.w * factor,
                                   quaternion.x * factor,
                                   quaternion.y * factor,
                                   quaternion.z * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator*(Value const factor, Quaternion3D<Value> const& quaternion)
    {
        try {
            return quaternion * factor;
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value>
    constexpr auto operator/(Quaternion3D<Value> const& quaternion, Value const factor)
    {
        try {
            return quaternion * (1 / factor);
        } catch (Error const& error) {
            throw error;
        }
    }

}; // namespace Linalg

#endif // QUATERNION3D_HPP