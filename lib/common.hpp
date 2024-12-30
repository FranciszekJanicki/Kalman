#ifndef COMMON_HPP
#define COMMON_HPP

#include <concepts>
#include <cstddef>
#include <exception>
#include <fmt/core.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace Linalg {

    template <typename Type>
    concept Arithmetic = std::is_arithmetic_v<Type>;

    using Size = std::size_t;
    using Error = std::runtime_error;

}; // namespace Linalg

#endif // COMMON_HPP