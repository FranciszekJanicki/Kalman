#ifndef QUATERNION3D_HPP
#define QUATERNION3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <cstdlib>
#include <tuple>
#include <utility>

namespace Linalg {

    template <Arithmetic Value>
    struct Quaternion3D {
        [[nodiscard]] constexpr Quaternion3D conjugated() const noexcept
        {
            return Quaternion3D{w, -x, -y, -z};
        }

        constexpr void conjugate() noexcept
        {
            x = -x;
            y = -y;
            z = -z;
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return std::sqrt(std::pow(w, 2) + std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
        }

        [[nodiscard]] constexpr Quaternion3D normalized() const noexcept
        {
            const auto im{static_cast<Value>(1) / magnitude()};
            return Quaternion3D{w * im, x * im, y * im, z * im};
        }

        constexpr void normalize() noexcept
        {
            const auto im{static_cast<Value>(1) / magnitude()};
            w *= im;
            x *= im;
            y *= im;
            z *= im;
        }

        constexpr Quaternion3D& operator+=(const Quaternion3D& other) noexcept
        {
            this->w += other.w;
            this->x += other.x;
            this->y += other.y;
            this->z += other.z;
            return *this;
        }

        constexpr Quaternion3D& operator-=(const Quaternion3D& other) noexcept
        {
            this->w -= other.w;
            this->x -= other.x;
            this->y -= other.y;
            this->z -= other.z;
            return *this;
        }

        constexpr Quaternion3D& operator*=(const Quaternion3D& other) noexcept
        {
            const auto& [left_w, left_x, left_y, left_z] = std::forward_as_tuple(this->w, this->x, this->y, this->z);
            const auto& [right_w, right_x, right_y, right_z] =
                std::forward_as_tuple(other.w, other.x, other.y, other.z);
            this->w = left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z;
            this->x = left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y;
            this->y = left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x;
            this->z = left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w;
            return *this;
        }

        [[nodiscard]] constexpr bool operator<=>(const Quaternion3D& other) const noexcept = default;

        Value w{};
        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(const Quaternion3D<Value>& left, const Quaternion3D<Value>& right) noexcept
    {
        return Quaternion3D<Value>{left.w + right.w, left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator-(const Quaternion3D<Value>& left, const Quaternion3D<Value>& right) noexcept
    {
        return Quaternion3D<Value>{left.w - right.w, left.x - right.x, left.y - right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Quaternion3D<Value>& left, const Quaternion3D<Value>& other) noexcept
    {
        const auto& [left_w, left_x, left_y, left_z] = std::forward_as_tuple(left.w, left.x, left.y, left.z);
        const auto& [right_w, right_x, right_y, right_z] = std::forward_as_tuple(other.w, other.x, other.y, other.z);
        return Quaternion3D<Value>{left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
                                   left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
                                   left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
                                   left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w};
    }
}; // namespace Linalg

#endif // QUATERNION3D_HPP