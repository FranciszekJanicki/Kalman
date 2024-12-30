#ifndef STACK_VECTOR_HPP
#define STACK_VECTOR_HPP

#include "common.hpp"
#include <array>
#include <cstddef>
#include <exception>
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

namespace Linalg {

    namespace Stack {

        template <Arithmetic Value, Size ELEMS>
        struct Vector {
            using Data = std::array<Value, ELEMS>;

            [[nodiscard]] constexpr Value const& operator[](this Vector const& self, Size const elem)
            {
                if (elem >= ELEMS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[elem];
            }

            [[nodiscard]] constexpr Value& operator[](this Vector& self, Size const elem)
            {
                if (elem >= ELEMS) {
                    throw Error{"Out of bounds\n"};
                }
                return self.data[elem];
            }

            [[nodiscard]] explicit constexpr operator Data(this Vector const& self) noexcept
            {
                return self.data;
            }

            [[nodiscard]] constexpr Size size(this Vector const& self) noexcept
            {
                return ELEMS;
            }

            [[nodiscard]] constexpr Vector& operator+=(this Vector& self, Vector const& other) noexcept
            {
                self = sum(self, other);
                return self;
            }

            [[nodiscard]] constexpr Vector& operator-=(this Vector& self, Vector const& other) noexcept
            {
                self = difference(self, other);
                return self;
            }

            [[nodiscard]] constexpr Vector& operator*=(this Vector& self, Value const scale)
            {
                try {
                    self = scale(self, scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            [[nodiscard]] constexpr Vector& operator/=(this Vector& self, Value const scale) noexcept
            {
                try {
                    self = scale(self, 1 / scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            Data data{};
        }; // namespace Stack

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] Vector vector_sum(Vector const& left, Vector const& right) noexcept
        {
            Vector sum;
            for (Size i{}; i < ELEMS; ++i) {
                sum[i] = left[i] + right[i];
            }
            return sum;
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] Vector<Value, ELEMS> vector_difference(Vector<Value, ELEMS> const& left,
                                                             Vector<Value, ELEMS> const& right) noexcept
        {
            Vector difference;
            for (Size i{}; i < ELEMS; ++i) {
                difference[i] = left[i] - right[i];
            }
            return difference;
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] Vector<Value, ELEMS> vector_scale(Vector<Value, ELEMS> const& vector,
                                                        Vector<Value, ELEMS> const scale)
        {
            if (scale == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf!\n"};
            }
            Vector scale;
            for (Size i{}; i < ELEMS; ++i) {
                scale[i] = vector[i] * scale;
            }
            return scale;
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] constexpr Vector<Value, ELEMS> operator+(Vector<Value, ELEMS> const& left,
                                                               Vector<Value, ELEMS> const& right) noexcept
        {
            return vector_sum(left, right);
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] constexpr Vector<Value, ELEMS> operator-(Vector<Value, ELEMS> const& left,
                                                               Vector<Value, ELEMS> const& right) noexcept
        {
            return vector_difference(left, right);
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] constexpr Vector<Value, ELEMS> operator*(Value const scale, Vector<Value, ELEMS> const& vector)
        {
            try {
                return vector_scale(vector, scale);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] constexpr Vector<Value, ELEMS> operator*(Vector<Value, ELEMS> const& vector, Value const scale)
        {
            try {
                return vector_scale(vector, scale);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <Arithmetic Value, Size ELEMS>
        [[nodiscard]] constexpr Vector<Value, ELEMS> operator/(Vector<Value, ELEMS> const& vector, Value const scale)
        {
            try {
                return vector_scale(vector, 1 / scale);
            } catch (Error const& error) {
                throw error;
            }
        }

    }; // namespace Stack
}; // namespace Linalg

#endif // STACK_VECTOR_HPP