#ifndef STACK_KALMAN_HPP
#define STACK_KALMAN_HPP

#include "common.hpp"
#include "stack_matrix.hpp"
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

namespace Linalg::Stack {

    template <Linalg::Arithmetic Value, Size STATES, Size INPUTS = 1UL, Size MEASUREMENTS = 1UL>
    struct Kalman {
    public:
        template <Size ROWS, Size COLS>
        using Matrix = Matrix<Value, ROWS, COLS>;

        constexpr Kalman(Matrix<STATES, 1> const& initial_state,
                         Matrix<STATES, STATES> const& state_transition,
                         Matrix<STATES, STATES> const& state_covariance,
                         Matrix<STATES, INPUTS> const& input_transition,
                         Matrix<INPUTS, INPUTS> const& input_covariance,
                         Matrix<MEASUREMENTS, STATES> const& measurement_transition,
                         Matrix<MEASUREMENTS, MEASUREMENTS> const& measurement_covariance,
                         Matrix<STATES, STATES> const& process_noise) :
            state_{initial_state},
            state_transition_{state_transition},
            state_covariance_{state_covariance},
            input_transition_{input_transition},
            input_covariance_{input_covariance},
            measurement_transition_{measurement_transition},
            measurement_covariance_{measurement_covariance},
            process_noise_{process_noise}
        {}

        [[nodiscard]] constexpr Matrix<STATES, 1>
        operator()(this Kalman& self, Matrix<1, INPUTS> const& input, Matrix<1, MEASUREMENTS> const& measurement)
        {
            try {
                self.predict(input);
                self.predicted_state_.print();
                self.correct(measurement);
                self.state_.print();
                return self.state_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

    private:
        constexpr void predict(this Kalman& self, Matrix<1, INPUTS> const& input)
        {
            try {
                self.predicted_state_ = (self.state_transition_ * self.state_) + (self.input_transition_ * input);

                self.predicted_state_covariance_ =
                    (self.state_transition_ * self.state_covariance_ * matrix_transpose(self.state_transition_)) +
                    self.process_noise_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        constexpr void correct(this Kalman& self, Matrix<1, MEASUREMENTS> const& measurement)
        {
            try {
                auto const innovation{measurement - (self.measurement_transition_ * self.predicted_state_)};

                auto const residual_covariance{(self.measurement_transition_ * self.predicted_state_covariance_ *
                                                matrix_transpose(self.measurement_transition_)) +
                                               self.measurement_covariance_};

                auto const kalman_gain{self.predicted_state_covariance_ *
                                       matrix_transpose(self.measurement_transition_) *
                                       matrix_inverse(residual_covariance)};

                self.state_ = self.predicted_state_ + (kalman_gain * innovation);

                self.state_covariance_ = (make_eye<Value, STATES>() - kalman_gain * self.measurement_transition_) *
                                         self.predicted_state_covariance_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        Matrix<STATES, 1UL> state_{};
        Matrix<STATES, 1UL> predicted_state_{};
        Matrix<STATES, STATES> state_transition_{};
        Matrix<STATES, STATES> state_covariance_{};
        Matrix<STATES, STATES> predicted_state_covariance_{};

        Matrix<STATES, INPUTS> input_transition_{};
        Matrix<INPUTS, INPUTS> input_covariance_{};

        Matrix<MEASUREMENTS, STATES> measurement_transition_{};
        Matrix<MEASUREMENTS, MEASUREMENTS> measurement_covariance_{};

        Matrix<STATES, STATES> process_noise_{};
    };

}; // namespace Linalg::Stack

#endif // STACK_KALMAN_HPP