#ifndef VECTOR6D_HPP
#define VECTOR6D_HPP

#include <utility>
#include "Vector6D.hpp"
#include "arithmetic.hpp"
#include <functional>
#include <compare>

namespace Linalg {

    template <Arithmetic Value>
    struct Vector6D {

        [[nodiscard]] constexpr auto distance(const Vector6D& other) const noexcept
        {
            return this->orientation.distance(other.orientation);
        }

        [[nodiscard]] constexpr auto magnitude() const noexcept
        {
            return this->orientation.magnitude() + this->position.magnitude();
        }

        [[nodiscard]] constexpr Vector6D normalized() const noexcept
        {
            const auto im{static_cast<Value>(1) / magnitude()};
            return Vector6D{position * im, orientation * im};
        }

        constexpr void normalize() noexcept
        {
            const auto im{static_cast<Value>(1) / magnitude()};
            position *= im;
            orientation *= im;
        }


        constexpr operator+=(const Vector6D& other) noexcept {
            this->position += other.position;
            this->orientation += other.orientation;   
        }

        constexpr operator-=(const Vector6D& other) noexcept {
                this->position -= other.position;
                this->orientation -= other.orientation;
        }

        constexpr operator*=(const Vector6D& other) noexcept {
                this->position *= other.position;
                this->orientation *= other.orientation;
        }
        constexpr operator*=(const Value factor) noexcept {
            this->position *= factor;
            this->operation *= factor;
        }

        constexpr operator/=(const Vector6D& other) noexcept {
                this->position /= other.position;
                this->orientation /= other.orientation;
        }
        constexpr operator/=(const Value factor) noexcept {
            this->position /= factor;
            this->operation /= factor;
        }

        [[nodiscard]] constexpr bool operator<=>(const Vector6D& other) const noexcept = default;


        Vector6D<Value> position{};
        Vector6D<Value> orientation{};
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
    constexpr auto operator/(const Vector6D<Value>& left, const Vector6D<Value>& right) noexcept
    {
        return Vector6D{left.position / right.position, left.orientation / right.orientation};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Value factor, const Vector6D<Value>& vector) noexcept
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation + right.orientation};
    }

    template <Arithmetic Value>
    constexpr auto operator*(const Vector6D<Value>& vector, const Value factor) noexcept
    {
        return Vector6D<Value>{vector.position * factor, vector.orientation * factor};
    }

    template <Arithmetic Value>
    constexpr auto operator/(const Vector6D<Value>& vector, const Value factor) noexcept
    {
        return Vector6D<Value>{vector.position / factor, vector.orientation / factor};
    }


} // Linalg;

#endif // VECTOR6D_HPP