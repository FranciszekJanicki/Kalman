#ifndef ADRC_HPP
#define ADRC_HPP

#include "common.hpp"

namespace Regulators {

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

}; // namespace Regulators

#endif // ADRC_HPP