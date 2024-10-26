#ifndef VECTOR6D_HPP
#define VECTOR6D_HPP

#include "arithmetic.hpp"
#include "vector3d.hpp"
#include <cmath>
#include <compare>
#include <fmt/core.h>
#include <functional>
#include <utility>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector6D {
        [[nodiscard]] constexpr auto distance(const Vector6D& other) const noexcept
        {
            return this->orientation.distance(other.orientation);
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return std::sqrt(std::pow(this->orientation.magnitude(), 2) + std::pow(this->position.magnitude(), 2));
        }

        [[nodiscard]] constexpr Vector6D normalized() const noexcept
        {
            return Vector6D{position.normalized(), orientation.normalized()};
        }

        constexpr void normalize() noexcept
        {
            position.normalize();
            orientation.normalize();
        }

        constexpr void print() const noexcept
        {
            position.print();
            orientation.print();
        }

        constexpr Vector6D& operator+=(const Vector6D& other) noexcept
        {
            this->position += other.position;
            this->orientation += other.orientation;
            return *this;
        }

        constexpr Vector6D& operator-=(const Vector6D& other) noexcept
        {
            this->position -= other.position;
            this->orientation -= other.orientation;
            return *this;
        }

        constexpr Vector6D& operator*=(const Vector6D& other) noexcept
        {
            this->position *= other.position;
            this->orientation *= other.orientation;
            return *this;
        }
        constexpr Vector6D& operator*=(const Value factor) noexcept
        {
            this->position *= factor;
            this->operation *= factor;
            return *this;
        }

        constexpr Vector6D& operator/=(const Vector6D& other) noexcept
        {
            this->position /= other.position;
            this->orientation /= other.orientation;
            return *this;
        }
        constexpr Vector6D& operator/=(const Value factor) noexcept
        {
            this->position /= factor;
            this->operation /= factor;
            return *this;
        }

        [[nodiscard]] constexpr bool operator<=>(const Vector6D& other) const noexcept = default;

        Vector3D<Value> position{};
        Vector3D<Value> orientation{};
    };

    template <Arithmetic Value>
    constexpr auto operator+(const Vector6D<Value>& left, const Vector6D<Value>& right) noexcept
    {
        return Vector6D<Value>{left.position + right.position, left.orientation + right.orientation};
    }

    template <Arithmetic Value>
    constexpr auto operator-(const Vector6D<Value>& left, const Vector6D<Value>& right) noexcept
    {
        return Vector6D<Value>{left.position + right.position, left.orientation + right.orientation};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Vector6D<Value>& left, const Vector6D<Value>& right) noexcept
    {
        return Vector6D<Value>{left.position * right.position, left.orientation * right.orientation};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Value factor, const Vector6D<Value>& vector) noexcept
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation * factor};
    }
    template <Arithmetic Value>
    constexpr auto operator*(const Vector6D<Value>& vector, const Value factor) noexcept
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator/(const Vector6D<Value>& left, const Vector6D<Value>& right) noexcept
    {
        return Vector6D{left.position / right.position, left.orientation / right.orientation};
    }
    template <Arithmetic Value>
    constexpr auto operator/(const Vector6D<Value>& vector, const Value factor) noexcept
    {
        return Vector6D<Value>{vector.position / factor, vector.orientation / factor};
    }

} // namespace Linalg

#endif // VECTOR6D_HPP