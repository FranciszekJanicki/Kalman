#ifndef ARITHMETIC_HPP
#define ARITHMETIC_HPP

#include <concepts>
#include <type_traits>

namespace lib {

    template <typename type>
    concept arithmetic = std::is_arithmetic_v<type>;

}; // namespace lib

#endif // ARITHMETIC_HPP