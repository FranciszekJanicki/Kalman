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

        constexpr Kalman(const FilterModel& filter_model, const MeasureModel& measure_model) :
            filter_model_{filter_model}, measure_model_{measure_model}
        {
            this->initialize();
        }

        constexpr Kalman(FilterModel&& filter_model, MeasureModel&& measure_model) noexcept :
            filter_model_{std::forward<FilterModel>(filter_model)},
            measure_model_{std::forward<MeasureModel>(measure_model)}
        {
            this->initialize();
        }

        constexpr void operator()(const Matrix& measurement)
        {
            this->predict();
            this->update(measurement);
        }

        [[nodiscard]] constexpr FilterModel&& filter_model() && noexcept
        {
            return std::forward<Kalman>(*this).filter_model_;
        }

        [[nodiscard]] constexpr const FilterModel& filter_model() const& noexcept
        {
            return this->filter_model_;
        }

        constexpr void filter_model(const FilterModel& filter_model) noexcept
        {
            this->filter_model_ = filter_model;
        }

        constexpr void filter_model(FilterModel&& filter_model) noexcept
        {
            this->filter_model_ = std::forward<FilterModel>(filter_model);
        }

        [[nodiscard]] constexpr MeasureModel&& measure_model() && noexcept
        {
            return std::forward<Kalman>(*this).measure_model_;
        }

        [[nodiscard]] constexpr const MeasureModel& measure_model() const& noexcept
        {
            return this->measure_model_;
        }

        constexpr void measure_model(const MeasureModel& measure_model) noexcept
        {
            this->measure_model_ = measure_model;
        }

        constexpr void measure_model(MeasureModel&& measure_model) noexcept
        {
            this->measure_model_ = std::forward<MeasureModel>(measure_model);
        }

        constexpr void print_state() const noexcept
        {
            this->filter_model_.state_.print();
        }

        constexpr void print_predicted() const noexcept
        {
            this->measure_model_.predicted_state_.print();
        }

    private:
        constexpr void predict()
        {
            if (!this->is_initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* predict state */
            this->predicted_state_ = this->filter_model_.state_transition * this->state_;

            /* predict covariance */
            this->filter_model_.state_covariance =
                ((this->filter_model_.state_transition * this->filter_model_.state_covariance) *
                 this->filter_model_.state_transition);
            this->filter_model_.state_covariance +=
                ((this->filter_model_.input_transition * this->filter_model_.input_covariance) *
                 this->filter_model_.input_transition);
        }

        constexpr void update(const Matrix& measurement)
        {
            if (!this->is_initialized_) {
                fmt::print("Filter uninitialized!");
                return;
            }

            /* calculate innovation */
            this->measure_model_.innovation =
                measurement - (this->measure_model_.measurement_transition * this->state_);

            /* calculate residual covariance */
            auto temp_measurement_transition{this->measure_model_.measurement_transition};
            temp_measurement_transition.transpose();
            this->measure_model_.residual_covariance =
                (this->measure_model_.measurement_transition * this->filter_model_.state_covariance *
                 temp_measurement_transition) +
                this->measure_model_.process_noise;

            /* calculate kalman gain */
            auto temp_residual_covariance{this->measure_model_.residual_covariance};
            temp_residual_covariance.invert();
            this->measure_model_.kalman_gain =
                ((this->filter_model_.state_covariance * temp_measurement_transition) * temp_residual_covariance);

            /* correct state prediction */
            this->state_ *= (this->measure_model_.kalman_gain * this->measure_model_.innovation);

            /* correct state covariance */
            this->filter_model_.state_covariance -=
                this->measure_model_.kalman_gain *
                (this->measure_model_.measurement_transition * this->filter_model_.state_covariance);
        }

        void initialize() noexcept
        {
            if (!this->is_initialized_) {
                fmt::print("Checking correct dimensions");
                assert(this->filter_model_.states == this->measure_model_.states);
                assert(this->filter_model_.state_transition.rows() == this->filter_model_.states);
                assert(this->filter_model_.state_transition.columns() == this->filter_model_.inputs);
                assert(this->filter_model_.input_transition.rows() == this->filter_model_.states);
                assert(this->filter_model_.input_transition.columns() == this->filter_model_.inputs);
                assert(this->filter_model_.state_covariance.rows() == this->filter_model_.states);
                assert(this->filter_model_.state_covariance.columns() == this->filter_model_.states);
                assert(this->filter_model_.input_covariance.rows() == this->filter_model_.inputs);
                assert(this->filter_model_.input_covariance.columns() == this->filter_model_.inputs);
                assert(this->measure_model_.measurement_transition.rows() == this->measure_model_.measurements);
                assert(this->measure_model_.measurement_transition.columns() == this->measure_model_.states);
                assert(this->measure_model_.process_noise.rows() == this->measure_model_.measurements);
                assert(this->measure_model_.process_noise.columns() == this->measure_model_.measurements);
                assert(this->measure_model_.innovation.rows() == this->measure_model_.measurements);
                assert(this->measure_model_.innovation.columns() == 1);
                assert(this->measure_model_.residual_covariance.rows() == this->measure_model_.measurements);
                assert(this->measure_model_.residual_covariance.columns() == this->measure_model_.measurements);
                assert(this->measure_model_.kalman_gain.rows() == this->measure_model_.states);
                assert(this->measure_model_.kalman_gain.columns() == this->measure_model_.measurements);
                fmt::print("Filter has correct dimmensions");
                this->is_initialized_ = true;
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