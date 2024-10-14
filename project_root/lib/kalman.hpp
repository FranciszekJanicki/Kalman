#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "matrix.hpp"

namespace lib {

    struct filter_model {
        std::size_t states{}; // number of filter units

        std::size_t inputs{}; // number of filter inputs

        matrix<float> A{}; // state transition matrix (numStates x numInputs)

        matrix<float> x{}; // state vector_data (numStates x 1)

        matrix<float> B{}; // input transition matrix (numStates x numInputs)

        matrix<float> u{}; // input vector_data (numInputs x 1)

        matrix<float> P{}; // state covariance matrix (numStates x numStates)

        matrix<float> Q{}; // input covariance matrix (numInputs x numInputs)
    };

    struct measure_model {
        std::size_t states{}; // number of filter units

        std::size_t measurements{}; // number of meauserements performed

        matrix<float> H{}; // measurement transformation matrix (measurements x states)

        matrix<float> z{}; // measurement vector_data (numMeasures x  1)

        matrix<float> R{}; // process noise (measurement uncertainty) (measurements x measurements)

        matrix<float> y{}; // innovation (measurements x 1)

        matrix<float> S{}; // residual covariance (measurements x measurements)

        matrix<float> K{}; // kalman gain (states x measurements)
    };

    class kalman {
    public:
        kalman(const filter_model& filter, const measure_model& measure);
        kalman(filter_model&& filter, measure_model&& measure) noexcept;

        [[nodiscard]] matrix<float>&& state() && noexcept;
        [[nodiscard]] const matrix<float>& state() const& noexcept;

        void inputs(const matrix<float>& inputs);
        void inputs(matrix<float>&& inputs) noexcept;

        void predict();
        void update();
        void print_state() const noexcept;
        void print_predicted() const noexcept;

    private:
        void initialize() noexcept;

        bool is_initialized_{false};

        [[maybe_unused]] float step_time_{1.0f};
        [[maybe_unused]] float current_time_{0.0f};
        [[maybe_unused]] float start_time_{0.0f};

        std::size_t states_{1};       // number of filter outputs
        std::size_t inputs_{1};       // number of filter inputs
        std::size_t measurements_{1}; // number of measerements performed

        // filter model
        matrix<float> A_{};  // state transition matrix (states_ x inputs_)
        matrix<float> B_{};  // input transition matrix (states_ x inputs_)
        matrix<float> u_{};  // input vector_data (inputs_ x 1)
        matrix<float> P_{};  // state covariance matrix (states_ x states_)
        matrix<float> Q_{};  // input covariance matrix (inputs_ x inputs_)
        matrix<float> x_{};  // state vector_data (states_ x 1)
        matrix<float> xP_{}; // predicted state vector_data (states_ x 1)

        ///////////////////////////////////////////////////

        matrix<float> H_{}; // measurement transformation matrix (measurements_ x states_)
        matrix<float> z_{}; // measurement vector_data (measurements_ x  1)
        matrix<float> R_{}; // process noise (measurement uncertainty) (measurements_ x measurements_)
        matrix<float> y_{}; // innovation vector_data (measurements_ x 1)
        matrix<float> S_{}; // residual covariance (measurements_ x measurements_)
        matrix<float> K_{}; // kalman gain (states_ x measurements_)

        ///////////////////////////////////////////////////

        // temporary objects for move semantics (operators taking rvalue refs)
        matrix<float> tempBQ_{};
        matrix<float> tempAP_{};
        matrix<float> tempHx_{};
        matrix<float> tempHP_{};
        matrix<float> tempH_{};
        matrix<float> tempPHt_{};
        matrix<float> tempS_{};
        matrix<float> tempKy_{};
    };

}; // namespace lib

#endif // KALMAN_HPP