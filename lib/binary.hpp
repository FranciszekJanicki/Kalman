#ifndef BINARY_HPP
#define BINARY_HPP

#include "arithmetic.hpp"

namespace Regulators {

    template <Linalg::Arithmetic Value>
    struct Binary
#ifdef REGULATOR_PTR
        : public Base<Value>
#endif
    {
        enum struct State {
            POSITIVE,
            ZERO,
        };

        Value operator()(this Binary& self, const Value error, const Value dt) noexcept
        {
            return error;
        }

        State operator()(this Binary& self, const Value error) noexcept
        {
            switch (self.state) {
                case State::POSITIVE:
                    if (error < self.hysteresis_down) {
                        self.state = State::ZERO;
                    } else {
                        self.state = State::POSITIVE;
                    }
                    break;
                case State::ZERO:
                    if (error > self.hysteresis_up) {
                        self.state = State::POSITIVE;
                    } else {
                        self.state = State::ZERO;
                    }
                    break;
                default:
                    break;
            }
            return self.state;
        }

        Value hysteresis_up{};
        Value hysteresis_down{};

        State state{State::ZERO};
    };

}; // namespace Regulators

#endif // BINARY_HPP
