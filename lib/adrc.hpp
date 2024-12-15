#ifndef ADRC_HPP
#define ADRC_HPP

#include "arithmetic.hpp"

namespace Regulator {

    template <Linalg::Arithmetic Value>
    struct ADRC
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        Value operator()(this ADRC& self, const Value error, const Value) noexcept
        {
            // implement adrc algorithm here
            return error;
        }
    };

}; // namespace Regulator

#endif // ADRC_HPP