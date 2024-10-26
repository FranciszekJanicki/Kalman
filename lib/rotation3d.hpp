#ifndef ROTATION3D_HPP
#define ROTATION3D_HPP

#include "arithmetic.hpp"
#include <array>
#include <cmath>
#include <compare>
#include <cstdlib>
#include <utility>
#include <vector3d.hpp>

namespace Linalg {

    template <Arithmetic Value>
    struct Rotation3D {
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
            const auto& [left_x, left_y, left_z] = std::forward_as_tuple(this->x, this->y, this->z);
            const auto& [right_x, right_y, right_z] = std::forward_as_tuple(other.x, other.y, other.z);
            this->x = Vector3D<Value>{left_x.x * right_x.x + left_x.y * right_y.x + left_x.z * right_z.x,
                                      left_x.x * right_x.y + left_x.y * right_y.y + left_x.z * right_z.y,
                                      left_x.x * right_x.z + left_x.y * right_y.z + left_x.z * right_z.z};
            this->y = Vector3D<Value>{left_y.x * right_x.x + left_y.y * right_y.x + left_y.z * right_z.x,
                                      left_y.x * right_x.y + left_y.y * right_y.y + left_y.z * right_z.y,
                                      left_y.x * right_x.z + left_y.y * right_y.z + left_y.z * right_z.z};
            this->z = Vector3D<Value>{left_z.x * right_x.x + left_z.y * right_y.x + left_z.z * right_z.x,
                                      left_z.x * right_x.y + left_z.y * right_y.y + left_z.z * right_z.y,
                                      left_z.x * right_x.z + left_z.y * right_y.z + left_z.z * right_z.z};
            return *this;
        }
        constexpr Rotation3D& operator*=(const Value factor) noexcept
        {
            this->x *= factor;
            this->y *= factor;
            this->z *= factor;
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
        const auto& [left_x, left_y, left_z] = std::forward_as_tuple(left.x, left.y, left.z);
        const auto& [right_x, right_y, right_z] = std::forward_as_tuple(right.x, right.y, right.z);
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
    constexpr auto operator*(const Value factor, const Rotation3D<Value>& matrix) noexcept
    {
        return Rotation3D<Value>{matrix.x * factor, matrix.y * factor, matrix.z * factor};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Rotation3D<Value>& matrix, const Value factor) noexcept
    {
        return Rotation3D<Value>{matrix.x * factor, matrix.y * factor, matrix.z * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator/(const Rotation3D<Value>& matrix, const Value factor) noexcept
    {
        return Rotation3D<Value>{matrix.x / factor, matrix.y / factor, matrix.z / factor};
    }

}; // namespace Linalg

#endif // ROTATION3D_HPP