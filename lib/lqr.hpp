#ifndef LQR_HPP
#define LQR_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <cassert>
#include <fmt/core.h>
#include <ranges>
#include <utility>
#include <vector>

namespace Regulator {

    template <Linalg::Arithmetic Value>
    struct LQR {
    public:
        using Matrix = Linalg::Matrix<Value>;
        using RicattiSolutions = std::vector<Matrix>;

        constexpr LQR(Matrix const& state_transition,
                      Matrix const& input_transition,
                      Matrix const& state_cost,
                      Matrix const& input_cost,
                      Matrix const& end_condition,
                      std::uint64_t const samples) :
            state_transition_{state_transition},
            input_transition_{input_transition},
            state_cost_{state_cost},
            input_cost_{input_cost},
            ricatti_solutions_{get_ricatti_solutions(state_transition,
                                                     input_transition,
                                                     state_cost,
                                                     input_cost,
                                                     end_condition,
                                                     samples)}
        {}

        constexpr LQR(Matrix&& state_transition,
                      Matrix&& input_transition,
                      Matrix&& state_cost,
                      Matrix&& input_cost,
                      Matrix const& end_condition,
                      std::uint64_t const samples) noexcept :
            state_transition_{std::forward<Matrix>(state_transition)},
            input_transition_{std::forward<Matrix>(input_transition)},
            state_cost_{std::forward<Matrix>(state_cost)},
            input_cost_{std::forward<Matrix>(input_cost)},
            ricatti_solutions_{get_ricatti_solutions(state_transition,
                                                     input_transition,
                                                     state_cost,
                                                     input_cost,
                                                     end_condition,
                                                     samples)}
        {}

        [[nodiscard]] constexpr Matrix
        operator()(this LQR& self, std::uint64_t const sample, Matrix const& input, Matrix const& measurement)
        {
            auto error{input - measurement};
            return input - (get_optimal_gain(sample,
                                             self.ricatti_solutions_[sample],
                                             self.input_transition_,
                                             self.input_cost_) *
                            error);
        }

    private:
        static constexpr Matrix get_optimal_gain(std::uint64_t const sample,
                                                 Matrix const& ricatti,
                                                 Matrix const& input_transition,
                                                 Matrix const& input_cost)
        {
            return Matrix::inverse(input_cost).value() * Matrix::transpose(input_transition) * ricatti;
        }

        static constexpr RicattiSolutions get_ricatti_solutions(Matrix const& state_transition,
                                                                Matrix const& input_transition,
                                                                Matrix const& state_cost,
                                                                Matrix const& input_cost,
                                                                Matrix const& end_condition,
                                                                std::uint64_t const samples)
        {
            RicattiSolutions solutions{};
            solutions.reserve(samples);
            solutions.push_back(end_condition);
            for (std::uint64_t i{}; i < samples; ++i) {
                Matrix solution{-1 * (solutions.back() * state_transition -
                                      solutions.back() * input_transition * Matrix::inverse(input_cost).value() *
                                          Matrix::transpose(input_transition) * solutions.back() +
                                      Matrix::transpose(state_transition) * solutions.back() + state_cost)};
                solutions.push_back(std::move(solution));
            }
            std::ranges::reverse(solutions);
            return solutions;
        }

        static constexpr Matrix get_ricatti_solution(Matrix const& state_transition,
                                                     Matrix const& input_transition,
                                                     Matrix const& state_cost,
                                                     Matrix const& input_cost,
                                                     Matrix const& start_condition,
                                                     Value const dt,
                                                     Value const tolerance)
        {
            Matrix prev_solution{start_condition};
            Value solution_error{std::numeric_limits<Value>::max()};
            while (solution_error > tolerance) {
                Matrix solution{-(prev_solution * state_transition -
                                  prev_solution * input_transition * Matrix::inverse(input_cost).value() *
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

        RicattiSolutions ricatti_solutions_{};
    };

}; // namespace Regulator

#endif // LQR_HPP