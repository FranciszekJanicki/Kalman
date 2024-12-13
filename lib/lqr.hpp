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
                      Matrix const& input_cost,
                      Value const dt,
                      Value const tolerance,
                      Matrix const start_condition) :
            state_transition_{state_transition},
            input_transition_{input_transition},
            state_cost_{state_cost},
            input_cost_{input_cost},
            optimal_gain_{get_optimal_gain(state_transition,
                                           input_transition,
                                           state_cost,
                                           input_cost,
                                           dt,
                                           tolerance,
                                           start_condition)}
        {}

        constexpr LQR(Matrix&& state_transition,
                      Matrix&& input_transition,
                      Matrix&& state_cost,
                      Matrix&& input_cost,
                      Value const dt,
                      Value const tolerance,
                      Matrix const start_condition) noexcept :
            state_transition_{std::forward<Matrix>(state_transition)},
            input_transition_{std::forward<Matrix>(input_transition)},
            state_cost_{std::forward<Matrix>(state_cost)},
            input_cost_{std::forward<Matrix>(input_cost)},
            optimal_gain_{get_optimal_gain(state_transition,
                                           input_transition,
                                           state_cost,
                                           input_cost,
                                           dt,
                                           tolerance,
                                           start_condition)}
        {}

        [[nodiscard]] constexpr Matrix operator()(this LQR& self, Matrix const& input, Matrix const& measurement)
        {
            auto error{input - measurement};
            return input - (self.optimal_gain_ * error);
        }

    private:
        static constexpr Matrix get_optimal_gain(Matrix const& state_transition,
                                                 Matrix const& input_transition,
                                                 Matrix const& state_cost,
                                                 Matrix const& input_cost,
                                                 Value const dt,
                                                 Value const tolerance,
                                                 Matrix const start_condition)
        {
            return Matrix::inverse(input_cost) * Matrix::transpose(input_transition) *
                   get_ricatti_solution(state_transition,
                                        input_transition,
                                        state_cost,
                                        input_cost,
                                        dt,
                                        tolerance,
                                        start_condition);
        }

        static constexpr Matrix get_ricatti_solution(Matrix const& state_transition,
                                                     Matrix const& input_transition,
                                                     Matrix const& state_cost,
                                                     Matrix const& input_cost,
                                                     Value const dt,
                                                     Value const tolerance,
                                                     Matrix const start_condition)
        {
            auto prev_solution{start_condition};
            auto solution_error{std::numeric_limits<Value>::max()};
            while (solution_error > tolerance) {
                auto solution{-(prev_solution * state_transition -
                                prev_solution * input_transition * Matrix::inverse(input_cost) *
                                    Matrix::transpose(input_transition) * prev_solution +
                                Matrix::transpose(state_transition) * prev_solution + state_cost) *
                              dt};

                solution_error = std::abs(solution - prev_solution);
                prev_solution = std::move(solution);
            }
            return prev_solution;
        }

        Matrix state_transition_{};
        Matrix input_transition_{};
        Matrix state_cost_{};
        Matrix input_cost_{};

        Matrix optimal_gain_{};
    };

}; // namespace Regulator

#endif // LQR_HPP