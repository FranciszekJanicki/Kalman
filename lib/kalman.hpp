#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "arithmetic.hpp"
#include "matrix.hpp"
#include <cassert>
#include <fmt/core.h>
#include <utility>

template <Linalg::Arithmetic Unit, Linalg::Arithmetic Time>
struct Kalman {
public:
    using Matrix = Linalg::Matrix<Unit>;

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
        initialize();
    }

    constexpr Kalman(FilterModel&& filter_model, MeasureModel&& measure_model) noexcept :
        filter_model_{std::forward<FilterModel>(filter_model)},
        measure_model_{std::forward<MeasureModel>(measure_model)}
    {
        initialize();
    }

    constexpr void operator()(const Matrix& measurement)
    {
        predict();
        update(measurement);
    }

    [[nodiscard]] constexpr FilterModel&& filter_model() && noexcept
    {
        return std::move(filter_model_);
    }

    [[nodiscard]] constexpr const FilterModel& filter_model() const& noexcept
    {
        return filter_model_;
    }

    constexpr void filter_model(const FilterModel& filter_model) noexcept
    {
        filter_model_ = filter_model;
    }

    constexpr void filter_model(FilterModel&& filter_model) noexcept
    {
        filter_model_ = std::forward<FilterModel>(filter_model);
    }

    [[nodiscard]] constexpr MeasureModel&& measure_model() && noexcept
    {
        return std::move(measure_model_);
    }

    [[nodiscard]] constexpr const MeasureModel& measure_model() const& noexcept
    {
        return measure_model_;
    }

    constexpr void measure_model(const MeasureModel& measure_model) noexcept
    {
        measure_model_ = measure_model;
    }

    constexpr void measure_model(MeasureModel&& measure_model) noexcept
    {
        measure_model_ = std::forward<MeasureModel>(measure_model);
    }

    constexpr void print_state() const noexcept
    {
        filter_model_.state_.print();
    }

    constexpr void print_predicted() const noexcept
    {
        measure_model_.predicted_state_.print();
    }

private:
    constexpr void predict()
    {
        if (!is_initialized_) {
            fmt::print("Filter uninitialized!");
            return;
        }

        /* predict state */
        predicted_state_ = filter_model_.state_transition * state_;

        /* predict covariance */
        filter_model_.state_covariance =
            ((filter_model_.state_transition * filter_model_.state_covariance) * filter_model_.state_transition);
        filter_model_.state_covariance +=
            ((filter_model_.input_transition * filter_model_.input_covariance) * filter_model_.input_transition);
    }

    constexpr void update(const Matrix& measurement)
    {
        if (!is_initialized_) {
            fmt::print("Filter uninitialized!");
            return;
        }

        /* calculate innovation */
        measure_model_.innovation = measurement - (measure_model_.measurement_transition * state_);

        /* calculate residual covariance */
        auto temp_measurement_transition{measure_model_.measurement_transition};
        temp_measurement_transition.transpose();
        measure_model_.residual_covariance =
            (measure_model_.measurement_transition * filter_model_.state_covariance * temp_measurement_transition) +
            measure_model_.process_noise;

        /* calculate kalman gain */
        auto temp_residual_covariance{measure_model_.residual_covariance};
        temp_residual_covariance.invert();
        measure_model_.kalman_gain =
            ((filter_model_.state_covariance * temp_measurement_transition) * temp_residual_covariance);

        /* correct state prediction */
        state_ *= (measure_model_.kalman_gain * measure_model_.innovation);

        /* correct state covariance */
        filter_model_.state_covariance -=
            measure_model_.kalman_gain * (measure_model_.measurement_transition * filter_model_.state_covariance);
    }

    void initialize() noexcept
    {
        if (!is_initialized_) {
            fmt::print("Checking correct dimensions");
            assert(filter_model_.states == measure_model_.states);
            assert(filter_model_.state_transition.rows() == filter_model_.states);
            assert(filter_model_.state_transition.columns() == filter_model_.inputs);
            assert(filter_model_.input_transition.rows() == filter_model_.states);
            assert(filter_model_.input_transition.columns() == filter_model_.inputs);
            assert(filter_model_.state_covariance.rows() == filter_model_.states);
            assert(filter_model_.state_covariance.columns() == filter_model_.states);
            assert(filter_model_.input_covariance.rows() == filter_model_.inputs);
            assert(filter_model_.input_covariance.columns() == filter_model_.inputs);
            assert(measure_model_.measurement_transition.rows() == measure_model_.measurements);
            assert(measure_model_.measurement_transition.columns() == measure_model_.states);
            assert(measure_model_.process_noise.rows() == measure_model_.measurements);
            assert(measure_model_.process_noise.columns() == measure_model_.measurements);
            assert(measure_model_.innovation.rows() == measure_model_.measurements);
            assert(measure_model_.innovation.columns() == 1);
            assert(measure_model_.residual_covariance.rows() == measure_model_.measurements);
            assert(measure_model_.residual_covariance.columns() == measure_model_.measurements);
            assert(measure_model_.kalman_gain.rows() == measure_model_.states);
            assert(measure_model_.kalman_gain.columns() == measure_model_.measurements);
            fmt::print("Filter has correct dimmensions");
            is_initialized_ = true;
        }
    }

    bool is_initialized_{false};

    [[maybe_unused]] Time step_time_{1};
    [[maybe_unused]] Time current_time_{0};
    [[maybe_unused]] Time start_time_{0};

    // calculated
    Matrix state_{};           // states x 1
    Matrix predicted_state_{}; // states x 1

    [[maybe_unused]] Matrix input_{};       // inputs x 1
    [[maybe_unused]] Matrix measurement_{}; // measurments x  1

    // constants
    FilterModel filter_model_{};
    MeasureModel measure_model_{};
};

#endif // KALMAN_HPP