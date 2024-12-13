#ifndef LQR_HPP
#define LQR_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <cassert>
#include <fmt/core.h>
#include <utility>

namespace Regulator {

    template <Linalg::Arithmetic Value>
    struct LQR {
    public:
        using Matrix = Linalg::Matrix<Value>;

        constexpr LQR(Matrix const& state_transition,
                      Matrix const& input_transition,
                      Matrix const& state_cost,
                      Matrix const& input_cost) :
            state_transition_{state_transition},
            input_transition_{input_transition},
            state_cost_{state_cost},
            input_cost_{input_cost}
        {}

        constexpr LQR(Matrix&& state_transition, Matrix&& input_transition, Matrix&& state_cost, Matrix&& input_cost) :
            state_transition_{std::forward<Matrix>(state_transition)},
            input_transition_{std::forward<Matrix>(input_transition)},
            state_cost_{std::forward<Matrix>(state_cost)},
            input_cost_{std::forward<Matrix>(input_cost)}
        {}

        [[nodiscard]] constexpr Matrix operator()(this LQR& self, Matrix const& input, Matrix const& measurement)
        {
            auto error{input - measurement};
            return input + (self.gain_ * error);
        }

    private:
        static constexpr Matrix get_ricatti_solution(Matrix const& state_transition,
                                                     Matrix const& input_transition,
                                                     Matrix const& state_cost,
                                                     Matrix const& input_cost)
        {
            return Matrix{};
        }

        Matrix state_transition_{};
        Matrix input_transition_{};
        Matrix state_cost_{};
        Matrix input_cost_{};

        Matrix gain_{Matrix::inverse(input_cost_) * Matrix::transpose(input_transition_) *
                     get_ricatti_solution(state_transition_, input_transition_, state_cost_, input_cost_)};
    };

}; // namespace Regulator

#endif // LQR_HPP