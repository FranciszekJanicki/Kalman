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

        struct FilterModel {
            std::size_t states{}; // number of filter units
            std::size_t inputs{}; // number of filter inputs

            Matrix state_transition{}; // states x inputs
            Matrix input_transition{}; // states x inputs
            Matrix state_covariance{}; // states x states
            Matrix input_covariance{}; //  inputs x inputs
        };

        struct MeasureModel {
            std::size_t states{};       // number of filter units
            std::size_t measurements{}; // number of meauserements performed

            Matrix measurement_transition{}; // measurements x states
            Matrix process_noise{};          // measurements x measurements
            Matrix innovation{};             // measurements x 1
            Matrix residual_covariance{};    // measurements x measurements
            Matrix kalman_gain{};            // states x measurements
        };

        constexpr Kalman(FilterModel const& filter_model, MeasureModel const& measure_model) :
            filter_model_{filter_model}, measure_model_{measure_model}
        {
            initialize();
        }

        constexpr Kalman(FilterModel&& filter_model, MeasureModel&& measure_model) noexcept :
            filter_model_{std::forward<FilterModel>(filter_model)},
            measure_model_{std::forward<MeasureModel>(measure_model)}
        {
            initialize();
        }

        constexpr void operator()(this Kalman& self, Matrix const& measurement)
        {
            self.predict();
            self.update(measurement);
        }

        [[nodiscard]] constexpr FilterModel&& filter_model(this Kalman&& self) noexcept
        {
            return std::forward<Kalman>(self).filter_model_;
        }

        [[nodiscard]] constexpr FilterModel const& filter_model(this Kalman const& self) noexcept
        {
            return self.filter_model_;
        }

        constexpr void filter_model(this Kalman& self, FilterModel const& filter_model) noexcept
        {
            self.filter_model_ = filter_model;
        }

        constexpr void filter_model(this Kalman& self, FilterModel&& filter_model) noexcept
        {
            self.filter_model_ = std::forward<FilterModel>(filter_model);
        }

        [[nodiscard]] constexpr MeasureModel&& measure_model(this Kalman&& self) noexcept
        {
            return std::forward<Kalman>(self).measure_model_;
        }

        [[nodiscard]] constexpr MeasureModel const& measure_model(this Kalman const& self) noexcept
        {
            return self.measure_model_;
        }

        constexpr void measure_model(this Kalman& self, MeasureModel const& measure_model) noexcept
        {
            self.measure_model_ = measure_model;
        }

        constexpr void measure_model(this Kalman& self, MeasureModel&& measure_model) noexcept
        {
            self.measure_model_ = std::forward<MeasureModel>(measure_model);
        }

        constexpr void print_state(this Kalman const& self) noexcept
        {
            self.filter_model_.state_.print();
        }

        constexpr void print_predicted(this Kalman const& self) noexcept
        {
            self.measure_model_.predicted_state_.print();
        }

    private:
        constexpr void predict(this Kalman& self)
        {
            if (!self.is_initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* predict state */
            self.predicted_state_ = self.filter_model_.state_transition * self.state_;

            /* predict covariance */
            self.filter_model_.state_covariance =
                ((self.filter_model_.state_transition * self.filter_model_.state_covariance) *
                 self.filter_model_.state_transition);
            self.filter_model_.state_covariance +=
                ((self.filter_model_.input_transition * self.filter_model_.input_covariance) *
                 self.filter_model_.input_transition);
        }

        constexpr void update(this Kalman& self, const Matrix& measurement)
        {
            if (!self.is_initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* calculate innovation */
            self.measure_model_.innovation = measurement - (self.measure_model_.measurement_transition * self.state_);

            /* calculate residual covariance */
            auto temp_measurement_transition{self.measure_model_.measurement_transition};
            temp_measurement_transition.transpose();
            self.measure_model_.residual_covariance =
                (self.measure_model_.measurement_transition * self.filter_model_.state_covariance *
                 temp_measurement_transition) +
                self.measure_model_.process_noise;

            /* calculate kalman gain */
            auto temp_residual_covariance{self.measure_model_.residual_covariance};
            temp_residual_covariance.invert();
            self.measure_model_.kalman_gain =
                ((self.filter_model_.state_covariance * temp_measurement_transition) * temp_residual_covariance);

            /* correct state prediction */
            self.state_ *= (self.measure_model_.kalman_gain * self.measure_model_.innovation);

            /* correct state covariance */
            self.filter_model_.state_covariance -=
                self.measure_model_.kalman_gain *
                (self.measure_model_.measurement_transition * self.filter_model_.state_covariance);
        }

        void initialize(this Kalman& self) noexcept
        {
            if (!self.is_initialized_) {
                fmt::print("Checking correct dimensions");
                assert(self.filter_model_.states == self.measure_model_.states);
                assert(self.filter_model_.state_transition.rows() == self.filter_model_.states);
                assert(self.filter_model_.state_transition.columns() == self.filter_model_.inputs);
                assert(self.filter_model_.input_transition.rows() == self.filter_model_.states);
                assert(self.filter_model_.input_transition.columns() == self.filter_model_.inputs);
                assert(self.filter_model_.state_covariance.rows() == self.filter_model_.states);
                assert(self.filter_model_.state_covariance.columns() == self.filter_model_.states);
                assert(self.filter_model_.input_covariance.rows() == self.filter_model_.inputs);
                assert(self.filter_model_.input_covariance.columns() == self.filter_model_.inputs);
                assert(self.measure_model_.measurement_transition.rows() == self.measure_model_.measurements);
                assert(self.measure_model_.measurement_transition.columns() == self.measure_model_.states);
                assert(self.measure_model_.process_noise.rows() == self.measure_model_.measurements);
                assert(self.measure_model_.process_noise.columns() == self.measure_model_.measurements);
                assert(self.measure_model_.innovation.rows() == self.measure_model_.measurements);
                assert(self.measure_model_.innovation.columns() == 1);
                assert(self.measure_model_.residual_covariance.rows() == self.measure_model_.measurements);
                assert(self.measure_model_.residual_covariance.columns() == self.measure_model_.measurements);
                assert(self.measure_model_.kalman_gain.rows() == self.measure_model_.states);
                assert(self.measure_model_.kalman_gain.columns() == self.measure_model_.measurements);
                fmt::print("Filter has correct dimmensions");
                self.is_initialized_ = true;
            }
        }

        bool is_initialized_{false};

        [[maybe_unused]] Value step_time_{1};
        [[maybe_unused]] Value current_time_{0};
        [[maybe_unused]] Value start_time_{0};

        // calculated
        Matrix state_{};           // states x 1
        Matrix predicted_state_{}; // states x 1

        [[maybe_unused]] Matrix input_{};       // inputs x 1
        [[maybe_unused]] Matrix measurement_{}; // measurments x  1

        // constants
        FilterModel filter_model_{};
        MeasureModel measure_model_{};
    };

}; // namespace Filter

#endif // KALMAN_HPP