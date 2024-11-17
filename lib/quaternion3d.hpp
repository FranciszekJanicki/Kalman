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
        [[nodiscard]] constexpr Quaternion3D conjugated(this Quaternion3D const& self) noexcept
        {
            return Quaternion3D{self.w, -self.x, -self.y, -self.z};
        }

        constexpr void conjugate(this Quaternion3D& self) noexcept
        {
            self = self.conjugated();
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
            self = self.normalized();
        }

        constexpr Quaternion3D& operator+=(this Quaternion3D& self, Quaternion3D const& other) noexcept
        {
            self.w += other.w;
            self.x += other.x;
            self.y += other.y;
            self.z += other.z;
            return self;
        }

        constexpr Quaternion3D& operator-=(this Quaternion3D& self, Quaternion3D const& other) noexcept
        {
            self.w -= other.w;
            self.x -= other.x;
            self.y -= other.y;
            self.z -= other.z;
            return self;
        }

        constexpr Quaternion3D& operator*=(this Quaternion3D& self, Quaternion3D const& other) noexcept
        {
            auto const& [left_w, left_x, left_y, left_z] = std::forward_as_tuple(self.w, self.x, self.y, self.z);
            auto const& [right_w, right_x, right_y, right_z] =
                std::forward_as_tuple(other.w, other.x, other.y, other.z);
            self.w = left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z;
            self.x = left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y;
            self.y = left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x;
            self.z = left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w;
            return self;
        }

        [[nodiscard]] constexpr bool operator<=>(this Quaternion3D const& self,
                                                 Quaternion3D const& other) noexcept = default;

        Value w{};
        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr Quaternion3D<Value> operator+(Quaternion3D<Value> const& left, Quaternion3D<Value> const& right) noexcept
    {
        return Quaternion3D<Value>{left.w + right.w, left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr Quaternion3D<Value> operator-(Quaternion3D<Value> const& left, Quaternion3D<Value> const& right) noexcept
    {
        return Quaternion3D<Value>{left.w - right.w, left.x - right.x, left.y - right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr Quaternion3D<Value> operator*(Quaternion3D<Value> const& left, Quaternion3D<Value> const& other) noexcept
    {
        auto const& [left_w, left_x, left_y, left_z] = std::forward_as_tuple(left.w, left.x, left.y, left.z);
        auto const& [right_w, right_x, right_y, right_z] = std::forward_as_tuple(other.w, other.x, other.y, other.z);
        return Quaternion3D<Value>{left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
                                   left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
                                   left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
                                   left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w};
    }
}; // namespace Linalg

#endif // QUATERNION3D_HPP