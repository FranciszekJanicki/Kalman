#ifndef VECTOR3D_HPP
#define VECTOR3D_HPP

#include "arithmetic.hpp"
#include <cmath>
#include <compare>
#include <exception>
#include <quaternion3d.hpp>
#include <stdexcept>
#include <utility>

using Error = std::runtime_error;

namespace Linalg {

    template <Arithmetic Value>
    struct Vector3D {
        [[nodiscard]] constexpr auto distance(this Vector3D const& self, Vector3D const& other) noexcept
        {
            return std::sqrt(std::pow(self.x - other.x, 2) + std::pow(self.y - other.y, 2) +
                             std::pow(self.z - other.z, 2));
        }

        [[nodiscard]] constexpr auto magnitude(this Vector3D const& self) noexcept
        {
            return std::sqrt(std::pow(self.x, 2) + std::pow(self.y, 2) + std::pow(self.z, 2));
        }

        [[nodiscard]] constexpr Vector3D rotated(this Vector3D const& self,
                                                 Quaternion3D<Value> const& quaternion) noexcept
        {
            Quaternion3D p(0, self.x, self.y, self.z);
            p *= quaternion;
            p *= quaternion.conjugated();
            return Vector3D{p.x, p.y, p.z};
        }

        constexpr void rotate(this Vector3D& self, Quaternion3D<Value> const& quaternion) noexcept
        {
            Quaternion3D p(0, self.x, self.y, self.z);
            p *= quaternion;
            p *= quaternion.conjugated();
            self.x = p.x;
            self.y = p.y;
            self.z = p.z;
        }

        [[nodiscard]] constexpr Vector3D normalized(this Vector3D const& self) noexcept
        {
            const auto im{Value{1} / self.magnitude()};
            return Vector3D{self.x * im, self.y * im, self.z * im};
        }

        constexpr void normalize(this Vector3D& self) noexcept
        {
            const auto im{Value{1} / self.magnitude()};
            self *= im;
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
            self += (-1 * other);
            return self;
        }

        constexpr Vector3D& operator*=(this Vector3D& self, Value const factor)
        {
            if (factor == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf\n"};
            }
            self.x *= factor;
            self.y *= factor;
            self.z *= factor;
            return self;
        }

        constexpr Vector3D& operator/=(this Vector3D& self, Value const factor)
        {
            try {
                self *= (1 / factor);
                return self;
            } catch (Error const& error) {
                throw error;
            }
        }

        template <Arithmetic Converted>
        [[nodiscard]] explicit constexpr operator Vector3D<Converted>(this Vector3D const& self) noexcept
        {
            return Vector3D<Converted>{static_cast<Converted>(self.x),
                                       static_cast<Converted>(self.y),
                                       static_cast<Converted>(self.z)};
        }

        [[nodiscard]] constexpr bool operator<=>(this Vector3D const& self, Vector3D const& other) noexcept = default;

        Value x{};
        Value y{};
        Value z{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x + right.x, left.y + right.y, left.z + right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator-(Vector3D<Value> const& left, Vector3D<Value> const& right) noexcept
    {
        return Vector3D<Value>{left.x - right.x, left.y - right.y, left.z - right.z};
    }

    template <Arithmetic Value>
    constexpr auto operator*(Value const factor, Vector3D<Value> const& vector) noexcept
    {
        if (factor == std::numeric_limits<Value>::max()) {
            throw Error{"Multiplication by inf\n"};
        }
        return Vector3D<Value>{vector.x * factor, vector.y * factor, vector.z * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator*(Vector3D<Value> const& vector, Value const factor)
    {
        try {
            return factor * vector;
        } catch (Error const& error) {
            throw error;
        }
    }

    template <Arithmetic Value>
    constexpr auto operator/(Vector3D<Value> const& vector, Value const factor)
    {
        try {
            return vector * (1 / factor);
        } catch (Error const& error) {
            throw error;
        }
    }

}; // namespace Linalg

#endif // VECTOR3D_HPP