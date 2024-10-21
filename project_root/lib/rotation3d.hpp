#ifndef ROTATION3D_HPP
#define ROTATION3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <cstdlib>
#include <fmt/core.h>
#include <utility>
#include <vector3d.hpp>

namespace Linalg {

    template <Arithmetic Value>
    struct Rotation3D {
        constexpr void print() const noexcept
        {
            x.print();
            y.print();
            z.print();
        }
        constexpr Rotation3D& operator+=(const Rotation3D& other) noexcept
        {
            this->x += other.x;
            this->y += other.y;
            this->z += other.z;
            return *this;
        }

        constexpr Rotation3D& operator-=(const Rotation3D& other) noexcept
        {
            this->x -= other.x;
            this->y -= other.y;
            this->z -= other.z;
            return *this;
        }

        constexpr Rotation3D& operator*=(const Rotation3D& other) noexcept
        {
            this->x *= other.x;
            this->y *= other.y;
            this->z *= other.z;
            return *this;
        }
        constexpr Rotation3D& operator*=(const Value factor) noexcept
        {
            this->x *= factor;
            this->y *= factor;
            this->z *= factor;
            return *this;
        }

        constexpr Rotation3D& operator/=(const Rotation3D& other) noexcept
        {
            this->x /= other.x;
            this->y /= other.y;
            this->z /= other.z;
            return *this;
        }
        constexpr Rotation3D& operator/=(const Value factor) noexcept
        {
            this->x /= factor;
            this->y /= factor;
            this->z /= factor;
            return *this;
        }

        [[nodiscard]] constexpr bool operator<=>(const Rotation3D& other) const noexcept = default;

        Rotation3D<Value> x{};
        Rotation3D<Value> y{};
        Rotation3D<Value> z{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(const Rotation3D<Value>& left, const Rotation3D<Value>& right) noexcept
    {
        return Rotation3D<Value>{left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator-(const Rotation3D<Value>& left, const Rotation3D<Value>& right) noexcept
    {
        return Rotation3D<Value>{left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Rotation3D<Value>& left, const Rotation3D<Value>& right) noexcept
    {
        return Rotation3D<Value>{left.x * right.x, left.y * right.y, left.z * right.z};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Value factor, const Rotation3D<Value>& vector) noexcept
    {
        return Rotation3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Rotation3D<Value>& vector, const Value factor) noexcept
    {
        return Rotation3D{vector.x + factor, vector.y + factor, vector.z + factor};
    }

    template <Arithmetic Value>
    constexpr auto operator/(const Rotation3D<Value>& left, const Rotation3D<Value>& right) noexcept
    {
        return Rotation3D{left.x / right.x, left.y / right.y, left.z / right.z};
    }
    template <Arithmetic Value>
    constexpr auto operator/(const Rotation3D<Value>& vector, const Value factor) noexcept
    {
        return Rotation3D<Value>{vector.x / factor, vector.y / factor, vector.z / factor};
    }

}; // namespace Linalg

#endif // ROTATION3D_HPP