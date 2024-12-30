#ifndef ROTATION3D_HPP
#define ROTATION3D_HPP

#include "common.hpp"
#include "vector3d.hpp"
#include <array>
#include <cmath>
#include <compare>
#include <cstdlib>
#include <utility>

namespace Linalg {

    template <Arithmetic Value>
    struct Rotation3D {
        constexpr Rotation3D& operator+=(this Rotation3D& self, Rotation3D const& other) noexcept
        {
            self.x += other.x;
            self.y += other.y;
            self.z += other.z;
            return self;
        }

        constexpr Rotation3D& operator-=(this Rotation3D& self, Rotation3D const& other) noexcept
        {
            self.x -= other.x;
            self.y -= other.y;
            self.z -= other.z;
            return self;
        }

        constexpr Rotation3D& operator*=(this Rotation3D& self, Rotation3D const& other) noexcept
        {
            self.x = Vector3D<Value>{self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x,
                                     self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y,
                                     self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z};
            self.y = Vector3D<Value>{self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x,
                                     self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y,
                                     self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z};
            self.z = Vector3D<Value>{self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x,
                                     self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y,
                                     self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z};
            return self;
        }

        constexpr Rotation3D& operator*=(this Rotation3D& self, Value const factor) noexcept
        {
            self.x *= factor;
            self.y *= factor;
            self.z *= factor;
            return self;
        }

        constexpr Rotation3D& operator/=(this Rotation3D& self, Value const factor) noexcept
        {
            self.x /= factor;
            self.y /= factor;
            self.z /= factor;
            return self;
        }

        [[nodiscard]] constexpr bool operator<=>(this Rotation3D const& self,
                                                 Rotation3D const& other) noexcept = default;

        Vector3D<Value> x{};
        Vector3D<Value> y{};
        Vector3D<Value> z{};
    };

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator+(Rotation3D<Value> const& left, Rotation3D<Value> const& right) noexcept
    {
        return Rotation3D<Value>{left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator-(Rotation3D<Value> const& left, Rotation3D<Value> const& right) noexcept
    {
        return Rotation3D<Value>{left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator*(Rotation3D<Value> const& left, Rotation3D<Value> const& right) noexcept
    {
        return Rotation3D<Value>{Vector3D<Value>{left.x * right.x + left.y * right.x + left.z * right.x,
                                                 left.x * right.y + left.y * right.y + left.z * right.y,
                                                 left.x * right.z + left.y * right.z + left.z * right.z},
                                 Vector3D<Value>{left.x * right.x + left.y * right.x + left.z * right.x,
                                                 left.x * right.y + left.y * right.y + left.z * right.y,
                                                 left.x * right.z + left.y * right.z + left.z * right.z},
                                 Vector3D<Value>{left.x * right.x + left.y * right.x + left.z * right.x,
                                                 left.x * right.y + left.y * right.y + left.z * right.y,
                                                 left.x * right.z + left.y * right.z + left.z * right.z}};
    }

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator*(Value const factor, Rotation3D<Value> const& matrix) noexcept
    {
        return Rotation3D<Value>{matrix.x * factor, matrix.y * factor, matrix.z * factor};
    }

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator*(Rotation3D<Value> const& matrix, Value const factor) noexcept
    {
        return Rotation3D<Value>{matrix.x * factor, matrix.y * factor, matrix.z * factor};
    }

    template <Arithmetic Value>
    constexpr Rotation3D<Value> operator/(Rotation3D<Value> const& matrix, Value const factor) noexcept
    {
        return Rotation3D<Value>{matrix.x / factor, matrix.y / factor, matrix.z / factor};
    }

}; // namespace Linalg

#endif // ROTATION3D_HPP