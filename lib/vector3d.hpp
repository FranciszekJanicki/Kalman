#ifndef VECTOR3D_HPP
#define VECTOR3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <quaternion3d.hpp>
#include <utility>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector3D {
        [[nodiscard]] constexpr auto distance(const Vector3D& other) const noexcept
        {
            return std::sqrt(std::pow(this->x - other.x, 2) + std::pow(this->y - other.y, 2) +
                             std::pow(this->z - other.z, 2));
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
        }

        [[nodiscard]] constexpr Vector3D rotated(const Quaternion3D<Value>& quaternion) const noexcept
        {
            Quaternion3D p(0, x, y, z);
            p *= quaternion;
            p *= quaternion.conjugated();
            return Vector3D{p.x, p.y, p.z};
        }

        constexpr void rotate(const Quaternion3D<Value>& q) noexcept
        {
            Quaternion3D p(0, x, y, z);
            p *= q;
            p *= q.conjugated();
            x = p.x;
            y = p.y;
            z = p.z;
        }

        [[nodiscard]] constexpr Vector3D normalized() const noexcept
        {
            const auto im{Value{1} / magnitude()};
            return Vector3D{x * im, y * im, z * im};
        }

        constexpr void normalize() noexcept
        {
            const auto im{Value{1} / magnitude()};
            x *= im;
            y *= im;
            z *= im;
        }

        constexpr Vector3D& operator+=(const Vector3D& other) noexcept
        {
            this->x += other.x;
            this->y += other.y;
            this->z += other.z;
            return *this;
        }

        constexpr Vector3D& operator-=(const Vector3D& other) noexcept
        {
            this->x -= other.x;
            this->y -= other.y;
            this->z -= other.z;
            return *this;
        }

        constexpr Vector3D& operator*=(const Vector3D& other) noexcept
        {
            this->x *= other.x;
            this->y *= other.y;
            this->z *= other.z;
            return *this;
        }
        constexpr Vector3D& operator*=(const Value factor) noexcept
        {
            this->x *= factor;
            this->y *= factor;
            this->z *= factor;
            return *this;
        }

        constexpr Vector3D& operator/=(const Vector3D& other) noexcept
        {
            this->x /= other.x;
            this->y /= other.y;
            this->z /= other.z;
            return *this;
        }
        constexpr Vector3D& operator/=(const Value factor) noexcept
        {
            this->x /= factor;
            this->y /= factor;
            this->z /= factor;
            return *this;
        }

        [[nodiscard]] constexpr bool operator<=>(const Vector3D& other) const noexcept = default;

        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(const Vector3D<Value>& left, const Vector3D<Value>& right) noexcept
    {
        return Vector3D<Value>{left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator-(const Vector3D<Value>& left, const Vector3D<Value>& right) noexcept
    {
        return Vector3D<Value>{left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Vector3D<Value>& left, const Vector3D<Value>& right) noexcept
    {
        return Vector3D<Value>{left.x * right.x, left.y * right.y, left.z * right.z};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Value factor, const Vector3D<Value>& vector) noexcept
    {
        return Vector3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Vector3D<Value>& vector, const Value factor) noexcept
    {
        return Vector3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator/(const Vector3D<Value>& left, const Vector3D<Value>& right) noexcept
    {
        return Vector3D<Value>{left.x / right.x, left.y / right.y, left.z / right.z};
    }
    template <Arithmetic Value>
    constexpr auto operator/(const Vector3D<Value>& vector, const Value factor) noexcept
    {
        return Vector3D<Value>{vector.x / factor, vector.y / factor, vector.z / factor};
    }

}; // namespace Linalg

#endif // VECTOR3D_HPP