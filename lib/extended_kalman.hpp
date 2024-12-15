#ifndef EXTENDED_KALMAN_HPP
#define EXTENDED_KALMAN_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

namespace Filter {

    template <Linalg::Arithmetic Value>
    struct ExtendedKalman {
    public:
        using Matrix = Linalg::Matrix<Value>;

        constexpr ExtendedKalman(Matrix&& initial_state,
                                 Matrix&& state_transition,
                                 Matrix&& state_covariance,
                                 Matrix&& input_transition,
                                 Matrix&& input_covariance,
                                 Matrix&& measurement_transition,
                                 Matrix&& measurement_covariance,
                                 Matrix&& process_noise) :
            state_{std::forward<Matrix>(initial_state)},
            state_transition_{std::forward<Matrix>(state_transition)},
            state_covariance_{std::forward<Matrix>(state_covariance)},
            input_transition_{std::forward<Matrix>(input_transition)},
            input_covariance_{std::forward<Matrix>(input_covariance)},
            measurement_transition_{std::forward<Matrix>(measurement_transition)},
            measurement_covariance_{std::forward<Matrix>(measurement_covariance)},
            process_noise_{std::forward<Matrix>(process_noise)}
        {
            initialize();
        }

        constexpr ExtendedKalman(Matrix const& initial_state,
                                 Matrix const& state_transition,
                                 Matrix const& state_covariance,
                                 Matrix const& input_transition,
                                 Matrix const& input_covariance,
                                 Matrix const& measurement_transition,
                                 Matrix const& measurement_covariance,
                                 Matrix const& process_noise) :
            state_{initial_state},
            state_transition_{state_transition},
            state_covariance_{state_covariance},
            input_transition_{input_transition},
            input_covariance_{input_covariance},
            measurement_transition_{measurement_transition},
            measurement_covariance_{measurement_covariance},
            process_noise_{process_noise}
        {
            initialize();
        }

        [[nodiscard]] constexpr Matrix
        operator()(this ExtendedKalman& self, Matrix const& input, Matrix const& measurement)
        {
            try {
                self.predict(input);
                self.correct(measurement);
                return self.state_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        constexpr void print_state(this ExtendedKalman const& self) noexcept
        {
            fmt::println("state: ");
            self.state_.print();

            fmt::println("predicted state: ");
            self.predicted_state_.print();
        }

        constexpr void print_covariance(this ExtendedKalman const& self) noexcept
        {
            fmt::println("state covariance: ");
            self.state_covariance_.print();

            fmt::println("predicted state covariance: ");
            self.predicted_state_covariance_.print();
        }

    private:
        constexpr void predict(this ExtendedKalman& self, Matrix const& input)
        {
            try {
                /* predict state */
                self.state_ = (self.state_transition_ * self.state_) + (self.input_transition_ * input);

                /* predict covariance_ */
                self.predicted_state_covariance_ =
                    (self.state_transition_ * self.state_covariance_ * Matrix::transpose(self.state_transition_)) +
                    self.process_noise_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        constexpr void correct(this ExtendedKalman& self, Matrix const& measurement)
        {
            try {
                /* calculate innovation_ */
                auto const innovation{measurement - (self.measurement_transition_ * self.state_)};

                /* calculate residual covariance_ */
                auto const residual_covariance{(self.measurement_transition_ * self.predicted_state_covariance_ *
                                                Matrix::transpose(self.measurement_transition_)) +
                                               self.measurement_covariance_};

                /* calculate kalman gain */
                auto const kalman_gain{
                    (self.predicted_state_covariance_ * Matrix::transpose(self.measurement_transition_)) *
                    Matrix::inverse(residual_covariance)};

                /* correct state prediction_ */
                self.state_ += (kalman_gain * innovation);

                /* correct state covariance_ */
                self.state_covariance_ =
                    (Matrix::make_eye(self.state_transition_.rows()) - kalman_gain * self.measurement_transition_) *
                    self.predicted_state_covariance_;
            } catch (std::runtime_error const& error) {
                throw error;
            }
        }

        void initialize(this ExtendedKalman& self)
        {
            auto const states{self.state_transition_.rows()};
            auto const inputs{self.input_transition_.columns()};
            auto const measurements{self.measurement_transition_.rows()};
            if (self.state_.rows() != states) {
                throw std::runtime_error{"Wrong state rows!"};
            }
            if (self.state_.columns() != 1) {
                throw std::runtime_error{"Wrong state columns!"};
            }
            if (self.state_transition_.rows() != states) {
                throw std::runtime_error{"Wrong state_transition rows!"};
            }
            if (self.state_transition_.columns() != states) {
                throw std::runtime_error{"Wrong state_transition columns!"};
            }
            if (self.input_transition_.rows() != states) {
                throw std::runtime_error{"Wrong input_transition rows!"};
            }
            if (self.input_transition_.columns() != inputs) {
                throw std::runtime_error{"Wrong input_transition columns!"};
            }
            if (self.state_covariance_.rows() != states) {
                throw std::runtime_error{"Wrong state_covariance rows!"};
            }
            if (self.state_covariance_.columns() != states) {
                throw std::runtime_error{"Wrong state_covariance columns!"};
            }
            if (self.input_covariance_.rows() != inputs) {
                throw std::runtime_error{"Wrong input_covariance rows!"};
            }
            if (self.input_covariance_.columns() != inputs) {
                throw std::runtime_error{"Wrong input_covariance columns!"};
            }
            if (self.measurement_transition_.rows() != measurements) {
                throw std::runtime_error{"Wrong measurement_transition rows!"};
            }
            if (self.measurement_transition_.columns() != states) {
                throw std::runtime_error{"Wrong measurement_transition columns!"};
            }
            if (self.measurement_covariance_.rows() != measurements) {
                throw std::runtime_error{"Wrong measurement_covariance rows!"};
            }
            if (self.measurement_covariance_.columns() != measurements) {
                throw std::runtime_error{"Wrong measurement_covariance columns!"};
            }
        }

        Matrix state_{}; // states x 1
        Matrix predicted_state_{};
        Matrix state_transition_{}; // states x inputs
        Matrix state_covariance_{}; // states x states
        Matrix predicted_state_covariance_{};

        Matrix input_transition_{}; // states x inputs
        Matrix input_covariance_{}; //  inputs x inputs

        Matrix measurement_transition_{}; // measurements x states
        Matrix measurement_covariance_{}; // measurements x measurements

        Matrix process_noise_{}; // states x states
    };

}; // namespace Filter

#endif // EXTENDED_KALMAN_HPP