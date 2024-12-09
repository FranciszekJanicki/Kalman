#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <cassert>
#include <fmt/core.h>
#include <utility>

namespace Filter {

    template <Linalg::Arithmetic Value>
    struct Kalman {
    public:
        using Matrix = Linalg::Matrix<Value>;

        constexpr Kalman(Matrix&& state_transition,
                         Matrix&& state_covariance,
                         Matrix&& input_transition,
                         Matrix&& input_covariance,
                         Matrix&& measurement_transition,
                         Matrix&& measurement_covariance) :
            state_transition_{std::forward<Matrix>(state_transition)},
            state_covariance_{std::forward<Matrix>(state_covariance)},
            input_transition_{std::forward<Matrix>(input_transition)},
            input_covariance_{std::forward<Matrix>(input_covariance)},
            measurement_transition_{std::forward<Matrix>(measurement_transition)},
            measurement_covariance_{std::forward<Matrix>(measurement_covariance)}
        {
            initialize();
        }

        constexpr Kalman(Matrix const& state_transition,
                         Matrix const& state_covariance,
                         Matrix const& input_transition,
                         Matrix const& input_covariance,
                         Matrix const& measurement_transition,
                         Matrix const& measurement_covariance) :
            state_transition_{state_transition},
            state_covariance_{state_covariance},
            input_transition_{input_transition},
            input_covariance_{input_covariance},
            measurement_transition_{measurement_transition},
            measurement_covariance_{measurement_covariance}
        {
            initialize();
        }

        [[nodiscard]] constexpr Matrix operator()(this Kalman& self, Matrix const& input, Matrix const& measurement)
        {
            self.predict(input);
            self.correct(measurement);
            return self.state_;
        }

        constexpr void print_state(this Kalman const& self) noexcept
        {
            self.state_.print();
        }

    private:
        constexpr void predict(this Kalman& self, Matrix const& input)
        {
            if (!self.initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* predict state */
            self.state_ = (self.state_transition_ * self.state_) + (self.input_transition_ * input);

            /* predict covariance_ */
            self.state_covariance_ =
                (self.state_transition_ * self.state_covariance_ * Matrix::transposition(self.state_transition_));
            self.state_covariance_ +=
                (self.input_transition_ * self.input_covariance_ * Matrix::transposition(self.input_transition_));
        }

        constexpr void correct(this Kalman& self, Matrix const& measurement)
        {
            if (!self.initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* calculate innovation_ */
            auto const innovation{measurement - (self.measurement_transition_ * self.state_)};

            /* calculate residual covariance_ */
            auto const residual_covariance{(self.measurement_transition_ * self.state_covariance_ *
                                            Matrix::transposition(self.measurement_transition_)) +
                                           self.measurement_covariance_};

            /* calculate kalman gain */
            auto const kalman_gain{(self.state_covariance_ * Matrix::transposition(self.measurement_transition_)) *
                                   Matrix::transposition(residual_covariance)};

            /* correct state prediction_ */
            self.state_ *= (kalman_gain * innovation);

            /* correct state covariance_ */
            self.state_covariance_ =
                (Matrix::make_eye(self.state_transition_.rows()) - kalman_gain * self.measurement_transition_) *
                self.state_covariance_;
        }

        void initialize(this Kalman& self) noexcept
        {
            if (!self.initialized_) {
                fmt::print("Checking correct dimensions");
                auto const states{self.state_transition_.rows()};
                auto const inputs{self.input_transition_.columns()};
                auto const measurements{self.measurement_transition_.rows()};
                assert(self.state_transition_.rows() == states);
                assert(self.state_transition_.columns() == inputs);
                assert(self.input_transition_.rows() == states);
                assert(self.input_transition_.columns() == inputs);
                assert(self.state_covariance_.rows() == states);
                assert(self.state_covariance_.columns() == states);
                assert(self.input_covariance_.rows() == inputs);
                assert(self.input_covariance_.columns() == inputs);
                assert(self.measurement_transition_.rows() == measurements);
                assert(self.measurement_transition_.columns() == states);
                assert(self.measurement_covariance_.rows() == measurements);
                assert(self.measurement_covariance_.columns() == measurements);
                fmt::print("Filter has correct dimmensions");
                self.initialized_ = true;
            }
        }

        bool initialized_{false};

        Matrix state_transition_{};       // states x inputs
        Matrix state_covariance_{};       // states x states
        Matrix input_transition_{};       // states x inputs
        Matrix input_covariance_{};       //  inputs x inputs
        Matrix measurement_transition_{}; // measurements x states
        Matrix measurement_covariance_{}; // measurements x measurements

        Matrix state_{Matrix::make_zeros(state_transition_.rows(), 1)}; // states x 1
    };

}; // namespace Filter

#endif // KALMAN_HPP