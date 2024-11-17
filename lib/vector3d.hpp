#ifndef VECTOR3D_HPP
#define VECTOR3D_HPP

#include "arithmetic.hpp"
#include "quaternion3d.hpp"
#include <cmath>
#include <compare>
#include <utility>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector3D {
        [[nodiscard]] constexpr auto distance(this Vector3D const& self, const Vector3D& other) noexcept
        {
            return std::sqrt(std::pow(self.x - other.x, 2) + std::pow(self.y - other.y, 2) +
                             std::pow(self.z - other.z, 2));
        }

        [[nodiscard]] constexpr auto magnitude(this Vector3D const& self) noexcept
        {
            return std::sqrt(std::pow(self.x, 2) + std::pow(self.y, 2) + std::pow(self.z, 2));
        }

        [[nodiscard]] constexpr Vector3D rotated(this Vector3D const& self,
                                                 const Quaternion3D<Value>& quaternion) noexcept
        {
            Quaternion3D p(0, self.x, self.y, self.z);
            p *= quaternion;
            p *= quaternion.conjugated();
            return Vector3D{p.x, p.y, p.z};
        }

        constexpr void rotate(this Vector3D& self, const Quaternion3D<Value>& quaternion) noexcept
        {
            self = self.rotated(quaternion);
        }

        [[nodiscard]] constexpr Vector3D normalized(this Vector3D const& self) noexcept
        {
            const auto im{Value{1} / self.magnitude()};
            return Vector3D{self.x * im, self.y * im, self.z * im};
        }

        constexpr void normalize(this Vector3D& self) noexcept
        {
            self = self.normalized();
        }

        constexpr Vector3D& operator+=(this Vector3D& self, Vector3D const& other) noexcept
        {
            self.x += other.x;
            self.y += other.y;
            self.z += other.z;
            return self;
        }

        constexpr Vector3D& operator-=(this Vector3D& self, Vector3D const& other) noexcept
        {
            self.x -= other.x;
            self.y -= other.y;
            self.z -= other.z;
            return self;
        }

        constexpr Vector3D& operator*=(this Vector3D& self, Vector3D const& other) noexcept
        {
            self.x *= other.x;
            self.y *= other.y;
            self.z *= other.z;
            return self;
        }
        constexpr Vector3D& operator*=(this Vector3D& self, Value const factor) noexcept
        {
            self.x *= factor;
            self.y *= factor;
            self.z *= factor;
            return self;
        }

        constexpr Vector3D& operator/=(this Vector3D& self, Vector3D const& other) noexcept
        {
            self.x /= other.x;
            self.y /= other.y;
            self.z /= other.z;
            return self;
        }
        constexpr Vector3D& operator/=(this Vector3D& self, Value const factor) noexcept
        {
            self.x /= factor;
            self.y /= factor;
            self.z /= factor;
            return self;
        }

        [[nodiscard]] constexpr bool operator<=>(this Vector3D const& self, Vector3D const& other) noexcept = default;

        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr Vector3D<Value> operator+(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr Vector3D<Value> operator-(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <Arithmetic Value>
    constexpr Vector3D<Value> operator*(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x * right.x, left.y * right.y, left.z * right.z};
    }
    template <Arithmetic Value>
    constexpr Vector3D<Value> operator*(Value const factor, Vector3D<Value> const& vector) noexcept
    {
        return Vector3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }
    template <Arithmetic Value>
    constexpr Vector3D<Value> operator*(Vector3D<Value> const& vector, Value const factor) noexcept
    {
        return Vector3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }

    template <Arithmetic Value>
    constexpr Vector3D<Value> operator/(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x / right.x, left.y / right.y, left.z / right.z};
    }
    template <Arithmetic Value>
    constexpr Vector3D<Value> operator/(Vector3D<Value> const& vector, Value const factor) noexcept
    {
        return Vector3D<Value>{vector.x / factor, vector.y / factor, vector.z / factor};
    }

}; // namespace Linalg

#endif // VECTOR3D_HPP