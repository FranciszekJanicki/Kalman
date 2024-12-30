#ifndef VECTOR6D_HPP
#define VECTOR6D_HPP

#include "common.hpp"
#include "quaternion3d.hpp"
#include "vector3d.hpp"
#include <cmath>
#include <compare>
#include <cstdlib>
#include <stdexcept>
#include <utility>

using Error = std::runtime_error;

namespace Linalg {

    template <Arithmetic Value>
    struct Vector6D {
        [[nodiscard]] constexpr auto distance(this Vector6D const& self, Vector6D const& other) noexcept
        {
            return self.orientation.distance(other.orientation);
        }

        [[nodiscard]] constexpr auto magnitude(this Vector6D const& self) noexcept
        {
            return std::sqrt(std::pow(self.orientation.magnitude(), 2) + std::pow(self.position.magnitude(), 2));
        }

        [[nodiscard]] constexpr Vector6D normalized(this Vector6D const& self) noexcept
        {
            return Vector6D{self.position.normalized(), self.orientation.normalized()};
        }

        constexpr void normalize(this Vector6D& self) noexcept
        {
            self.position.normalize();
            self.orientation.normalize();
        }

        constexpr Vector6D& operator+=(this Vector6D& self, Vector6D const& other) noexcept
        {
            self.position += other.position;
            self.orientation += other.orientation;
            return self;
        }

        constexpr Vector6D& operator-=(this Vector6D& self, Vector6D const& other) noexcept
        {
            self.position -= other.position;
            self.orientation -= other.orientation;
            return self;
        }

        constexpr Vector6D& operator*=(this Vector6D& self, Value const factor)
        {
            self.position *= factor;
            self.operation *= factor;
            return self;
        }

        constexpr Vector6D& operator/=(this Vector6D& self, Value const factor)
        {
            self.position /= factor;
            self.operation /= factor;
            return self;
        }

        [[nodiscard]] constexpr bool operator<=>(this Vector6D const& self, Vector6D const& other) noexcept = default;

        Vector3D<Value> position{};
        Vector3D<Value> orientation{};
    };

    template <Arithmetic Value>
    constexpr Vector6D<Value> operator+(Vector6D<Value> const& left, Vector6D<Value> const& right) noexcept
    {
        return Vector6D<Value>{left.position + right.position, left.orientation + right.orientation};
    }

    template <Arithmetic Value>
    constexpr Vector6D<Value> operator-(Vector6D<Value> const& left, Vector6D<Value> const& right) noexcept
    {
        return Vector6D<Value>{left.position + right.position, left.orientation + right.orientation};
    }

    template <Arithmetic Value>
    constexpr Vector6D<Value> operator*(Value const factor, Vector6D<Value> const& vector)
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation * factor};
    }

    template <Arithmetic Value>
    constexpr Vector6D<Value> operator*(Vector6D<Value> const& vector, Value const factor)
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation * factor};
    }

    template <Arithmetic Value>
    constexpr Vector6D<Value> operator/(Vector6D<Value> const& vector, Value const factor)
    {
        return Vector6D<Value>{vector.position / factor, vector.orientation / factor};
    }

} // namespace Linalg

#endif // VECTOR6D_HPP