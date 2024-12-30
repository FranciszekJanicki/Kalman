#ifndef HEAP_VECTOR_HPP
#define HEAP_VECTOR_HPP

#include "common.hpp"
#include <cassert>
#include <cmath>
#include <compare>
#include <exception>
#include <fmt/core.h>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Linalg {

    namespace Heap {

        template <Arithmetic Value>
        struct Vector {
        public:
            using Data = std::vector<Value>;
            using Init = std::initializer_list<Value>;

            [[nodiscard]] static Data make_data(Init const init)
            {
                // Data data{};

                // data.reserve(init.size());
                // for (auto const elem : init) {
                //     data.push_back(elem);
                // }

                // return data;
                return Data{init};
            }

            [[nodiscard]] static Data make_data(Size const elems)
            {
                // Data data{};

                // data.reserve(elems);
                // for (Size i{}; i < elems; ++i) {
                //     data.emplace_back();
                // }

                // return data;
                return Data(elems, Value{0});
            }

            [[nodiscard]] Vector zeros(Size const elems)
            {
                // return Vector{make_data(elems)};
                return Data(elems, Value{0});
            }

            [[nodiscard]] Vector ones(Size const elems)
            {
                // Vector result{make_data(elems)};

                // for (auto& elem : result) {
                //     elem = Value{1};
                // }

                // return result;
                return Data(elems, Value{1});
            }

            constexpr Vector() noexcept = default;

            constexpr Vector(Init const init) : data{data}
            {}

            constexpr Vector(Size const elems) : data{make_data(elems)}
            {}

            constexpr Vector(Vector const& other) = default;

            constexpr Vector(Vector&& other) noexcept = default;

            constexpr ~Vector() noexcept = default;

            constexpr Vector& operator=(Vector const& other) = default;

            constexpr Vector& operator=(Vector&& other) noexcept = default;

            constexpr void operator=(this Vector& self, Init const data)
            {
                self.data = data;
            }

            constexpr Vector& operator+=(this Vector& self, Vector const& other)
            {
                try {
                    self = vector_sum(self, other);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            constexpr Vector& operator-=(this Vector& self, Vector const& other)
            {
                try {
                    self = vector_difference(self, other);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            constexpr Vector& operator*=(this Vector& self, Value const scale)
            {
                try {
                    self = vector_scale(self, scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            constexpr Vector& operator/=(this Vector& self, Value const scale)
            {
                try {
                    self = vector_scale(self, 1 / scale);
                    return self;
                } catch (Error const& error) {
                    throw error;
                }
            }

            [[nodiscard]] constexpr Value& operator[](this Vector& self, Size const elem)
            {
                if (elem > self.elems()) {
                    throw Error{"Wrong dimensions\n"};
                }
                return self.data[elem];
            }

            [[nodiscard]] constexpr Value const& operator[](this Vector const& self, Size const elem) noexcept
            {
                if (elem > self.elems()) {
                    throw Error{"Wrong dimensions\n"};
                }
                return self.data[elem];
            }

            [[nodiscard]] constexpr bool operator<=>(this Vector const& self, Vector const& other) noexcept = default;

            constexpr void print(this Vector const& self) noexcept
            {
                fmt::print("[");
                for (auto& elem : self.data) {
                    fmt::print("{}", elem);
                    if (elem != self.data.back()) {
                        fmt::print(", ");
                    }
                }
                fmt::print("]\n");
            }

            [[nodiscard]] constexpr Size elems(this Vector const& self) noexcept
            {
                return self.data.size();
            }

            Data data{};
        };

        template <typename Value>
        [[nodiscard]] Vector<Value> vector_sum(Vector<Value> const& left, Vector<Value> const& right)
        {
            if (left.elems() != right.elems()) {
                throw Error{"Incorrect dimensions!\n"};
            }

            auto result{Vector<Value>::make_zeros(left.elems())};
            for (Size i{0}; i < left.elems(); ++i) {
                result[i] = left[i] + right[i];
            }
            return result;
        }

        template <typename Value>
        [[nodiscard]] Vector<Value> vector_difference(Vector<Value> const& left, Vector<Value> const& right)
        {
            if (left.elems() != right.elems()) {
                throw Error{"Incorrect dimensions!\n"};
            }

            auto result{Vector<Value>::make_zeros(left.elems())};
            for (Size i{0}; i < left.elems(); ++i) {
                result[i] = left[i] - right[i];
            }
            return result;
        }

        template <typename Value>
        [[nodiscard]] Vector<Value> vector_scale(Vector<Value> const& vector, Value const scale)
        {
            if (scale == std::numeric_limits<Value>::max()) {
                throw Error{"Multiplication by inf!\n"};
            }

            auto result{Vector<Value>::make_zeros(vector.elems())};
            for (Size i{0}; i < vector.elems(); ++i) {
                result[i] = vector[i] * scale;
            }
            return result;
        }

        template <typename Value>
        Vector<Value> operator+(Vector<Value> const& left, Vector<Value> const& right)
        {
            try {
                return vector_sum(left, right);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <typename Value>
        Vector<Value> operator-(Vector<Value> const& left, Vector<Value> const& right)
        {
            try {
                return vector_difference(left, right);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <typename Value>
        Vector<Value> operator*(Value const scale, Vector<Value> const& vector)
        {
            try {
                return vector_scale(vector, scale);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <typename Value>
        Vector<Value> operator*(Vector<Value> const& vector, Value const scale)
        {
            try {
                return vector_scale(vector, scale);
            } catch (Error const& error) {
                throw error;
            }
        }

        template <typename Value>
        Vector<Value> operator/(Vector<Value> const& vector, Value const scale)
        {
            try {
                return vector_scale(vector, 1 / scale);
            } catch (Error const& error) {
                throw error;
            }
        }

    }; // namespace Heap

}; // namespace Linalg

#endif // HEAP_VECTOR_HPP