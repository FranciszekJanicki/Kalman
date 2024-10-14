#ifndef QUATERNION_HPP
#define QUATERNION_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <cstdlib>
#include <fmt/core.h>
#include <tuple>
#include <utility>

namespace lib {

    template <arithmetic value_type>
    struct quaternion {
        [[nodiscard]] constexpr quaternion conjugate() const noexcept
        {
            return quaternion{w, -x, -y, -z};
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return std::sqrt(std::pow(w, 2) + std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
        }

        [[nodiscard]] constexpr quaternion normalize() const noexcept
        {
            const auto im{static_cast<value_type>(1) / magnitude()};
            return quaternion{w * im, x * im, y * im, z * im};
        }

        constexpr void print() const noexcept
        {
            fmt::print("a: {}, b: {}, c: {}, w: {}\n", x, y, z, w);
        }

        constexpr quaternion& operator+=(const quaternion& other) noexcept
        {
            this->w += other.w;
            this->x += other.x;
            this->y += other.y;
            this->z += other.z;
            return *this;
        }

        constexpr quaternion& operator-=(const quaternion& other) noexcept
        {
            this->w -= other.w;
            this->x -= other.x;
            this->y -= other.y;
            this->z -= other.z;
            return *this;
        }

        constexpr quaternion& operator*=(const quaternion& other) noexcept
        {
            auto [left_w, left_x, left_y, left_z] = std::forward_as_tuple(this->w, this->x, this->y, this->z);
            auto [right_w, right_x, right_y, right_z] = std::forward_as_tuple(other.w, other.x, other.y, other.z);
            this->w = left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z;
            this->x = left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y;
            this->y = left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x;
            this->z = left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w;
            return *this;
        }

        friend constexpr quaternion operator+(const quaternion& left, const quaternion& right) noexcept
        {
            return quaternion{left.w + right.w, left.x + right.x, left.y + right.y, left.z + right.z};
        }

        friend constexpr quaternion operator-(const quaternion& left, const quaternion& right) noexcept
        {
            return quaternion{left.w - right.w, left.x - right.x, left.y - right.y, left.z + right.z};
        }

        friend constexpr quaternion operator*(const quaternion& left, const quaternion& other) noexcept
        {
            auto [left_w, left_x, left_y, left_z] = std::forward_as_tuple(left.w, left.x, left.y, left.z);
            auto [right_w, right_x, right_y, right_z] = std::forward_as_tuple(other.w, other.x, other.y, other.z);
            return quaternion{left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
                              left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
                              left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
                              left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w};
        }

        [[nodiscard]] constexpr bool operator<=>(const quaternion& other) const noexcept = default;

        value_type w{};
        value_type x{};
        value_type y{};
        value_type z{};
    };

}; // namespace lib

#endif // QUATERNION_HPP