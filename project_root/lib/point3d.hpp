#ifndef POINT3D_HPP
#define POINT3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <fmt/core.h>
#include <quaternion.hpp>
#include <utility>

namespace lib {
    template <arithmetic value_type>
    struct point3D {
        [[nodiscard]] constexpr auto distance(const point3D& other) const noexcept
        {
            return std::sqrt(std::pow(this->x - other.x, 2) + std::pow(this->y - other.y, 2) +
                             std::pow(this->z - other.z, 2));
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
        }

        [[nodiscard]] constexpr point3D rotate(const quaternion<value_type>& q) const noexcept
        {
            quaternion p(0, x, y, z);
            p *= q;
            p *= q.conjugate();
            return point3D{p.x, p.y, p.z};
        }

        [[nodiscard]] constexpr point3D normalize() const noexcept
        {
            const auto im{static_cast<value_type>(1) / magnitude()};
            return point3D{x * im, y * im, z * im};
        }

        constexpr void print() const noexcept
        {
            fmt::print("a: {}, b: {}, c: {}, w: {}\n", x, y, z);
        }

        constexpr point3D& operator+=(const point3D& other) noexcept
        {
            this->x += other.x;
            this->y += other.y;
            this->z += other.z;
            return *this;
        }

        constexpr point3D& operator-=(const point3D& other) noexcept
        {
            this->x -= other.x;
            this->y -= other.y;
            this->z -= other.z;
            return *this;
        }

        constexpr point3D& operator*=(const point3D& other) noexcept
        {
            this->x *= other.x;
            this->y *= other.y;
            this->z *= other.z;
            return *this;
        }

        constexpr point3D& operator/=(const point3D& other) noexcept
        {
            this->x /= other.x;
            this->y /= other.y;
            this->z /= other.z;
            return *this;
        }

        constexpr point3D& operator*=(const value_type factor) noexcept
        {
            this->x *= factor;
            this->y *= factor;
            this->z *= factor;
            return *this;
        }

        constexpr point3D& operator/=(const value_type factor) noexcept
        {
            this->x /= factor;
            this->y /= factor;
            this->z /= factor;
            return *this;
        }

        friend constexpr point3D operator+(const point3D& left, const point3D& right) noexcept
        {
            return point3D{left.x += right.x, left.y += right.y, left.z += right.z};
        }

        friend constexpr point3D operator-(const point3D& left, const point3D& right) noexcept
        {
            return point3D{left.x -= right.x, left.y -= right.y, left.z -= right.z};
        }

        friend constexpr point3D operator*(const point3D& left, const point3D& right) noexcept
        {
            return point3D{left.x *= right.x, left.y *= right.y, left.z *= right.z};
        }

        friend constexpr point3D operator/(const point3D& left, const point3D& right) noexcept
        {
            return point3D{left.x /= right.x, left.y /= right.y, left.z /= right.z};
        }

        friend constexpr point3D operator*(const value_type factor, const point3D& point) noexcept
        {
            return point3D{point.x *= factor, point.y *= factor, point.z *= factor};
        }

        friend constexpr point3D operator*(const point3D& point, const value_type factor) noexcept
        {
            return point3D{point.x += factor, point.y += factor, point.z += factor};
        }

        friend constexpr point3D operator/(const point3D& point, const value_type factor) noexcept
        {
            return point3D{point.x /= factor, point.y /= factor, point.z /= factor};
        }

        [[nodiscard]] constexpr bool operator<=>(const point3D& other) const noexcept = default;

        enum class dim3D {
            x,
            y,
            z,
        };

        template <dim3D dim_choice>
        constexpr void operator++() noexcept
        {
            switch (dim_choice) {
                case dim3D::x:
                    x += step;
                case dim3D::y:
                    y += step;
                case dim3D::z:
                    z += step;
            }
        }

        template <dim3D dim_choice>
        constexpr void operator--() noexcept
        {
            switch (dim_choice) {
                case dim3D::x:
                    x -= step;
                case dim3D::y:
                    y -= step;
                case dim3D::z:
                    z -= step;
            }
        }

        static constexpr value_type step{};

        value_type x{};
        value_type y{};
        value_type z{};
    };

}; // namespace lib

#endif // POINT3D_HPP