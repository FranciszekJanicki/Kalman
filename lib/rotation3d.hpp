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
            auto const& [left_x, left_y, left_z] = std::forward_as_tuple(self.x, self.y, self.z);
            auto const& [right_x, right_y, right_z] = std::forward_as_tuple(other.x, other.y, other.z);
            self.x = Vector3D<Value>{left_x.x * right_x.x + left_x.y * right_y.x + left_x.z * right_z.x,
                                     left_x.x * right_x.y + left_x.y * right_y.y + left_x.z * right_z.y,
                                     left_x.x * right_x.z + left_x.y * right_y.z + left_x.z * right_z.z};
            self.y = Vector3D<Value>{left_y.x * right_x.x + left_y.y * right_y.x + left_y.z * right_z.x,
                                     left_y.x * right_x.y + left_y.y * right_y.y + left_y.z * right_z.y,
                                     left_y.x * right_x.z + left_y.y * right_y.z + left_y.z * right_z.z};
            self.z = Vector3D<Value>{left_z.x * right_x.x + left_z.y * right_y.x + left_z.z * right_z.x,
                                     left_z.x * right_x.y + left_z.y * right_y.y + left_z.z * right_z.y,
                                     left_z.x * right_x.z + left_z.y * right_y.z + left_z.z * right_z.z};
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
        auto const& [left_x, left_y, left_z] = std::forward_as_tuple(left.x, left.y, left.z);
        auto const& [right_x, right_y, right_z] = std::forward_as_tuple(right.x, right.y, right.z);
        return Rotation3D<Value>{Vector3D<Value>{left_x.x * right_x.x + left_x.y * right_y.x + left_x.z * right_z.x,
                                                 left_x.x * right_x.y + left_x.y * right_y.y + left_x.z * right_z.y,
                                                 left_x.x * right_x.z + left_x.y * right_y.z + left_x.z * right_z.z},
                                 Vector3D<Value>{left_y.x * right_x.x + left_y.y * right_y.x + left_y.z * right_z.x,
                                                 left_y.x * right_x.y + left_y.y * right_y.y + left_y.z * right_z.y,
                                                 left_y.x * right_x.z + left_y.y * right_y.z + left_y.z * right_z.z},
                                 Vector3D<Value>{left_z.x * right_x.x + left_z.y * right_y.x + left_z.z * right_z.x,
                                                 left_z.x * right_x.y + left_z.y * right_y.y + left_z.z * right_z.y,
                                                 left_z.x * right_x.z + left_z.y * right_y.z + left_z.z * right_z.z}};
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