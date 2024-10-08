#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "matrix_wrapper.hpp"

struct filter_model {
    std::size_t states{}; // number of filter units

    std::size_t inputs{}; // number of filter inputs

    matrix_wrapper<float> A{}; // state transition matrix (numStates x numInputs)

    matrix_wrapper<float> x{}; // state vector (numStates x 1)

    matrix_wrapper<float> B{}; // input transition matrix (numStates x numInputs)

    matrix_wrapper<float> u{}; // input vector (numInputs x 1)

    matrix_wrapper<float> P{}; // state covariance matrix (numStates x numStates)

    matrix_wrapper<float> Q{}; // input covariance matrix (numInputs x numInputs)
};

struct measure_model {
    std::size_t states{}; // number of filter units

    std::size_t measurements{}; // number of meauserements performed

    matrix_wrapper<float> H{}; // measurement transformation matrix (measurements x states)

    matrix_wrapper<float> z{}; // measurement vector (numMeasures x  1)

    matrix_wrapper<float> R{}; // process noise (measurement uncertainty) (measurements x measurements)

    matrix_wrapper<float> y{}; // innovation (measurements x 1)

    matrix_wrapper<float> S{}; // residual covariance (measurements x measurements)

    matrix_wrapper<float> K{}; // kalman gain (states x measurements)
};

class kalman {
public:
    kalman(const filter_model& filter, const measure_model& measure);

    kalman(filter_model&& filter, measure_model&& measure) noexcept;

    [[nodiscard]] matrix_wrapper<float>&& state() && noexcept;

    [[nodiscard]] const matrix_wrapper<float>& state() const& noexcept;

    void inputs(const matrix_wrapper<float>& inputs);

    void inputs(matrix_wrapper<float>&& inputs) noexcept;

    void predict();

    void update();

private:
    bool is_initialized_{false};

    [[maybe_unused]] float step_time_{1.0f};

    [[maybe_unused]] float current_time_{0.0f};

    [[maybe_unused]] float start_time_{0.0f};

    // common model
    size_t states_{1}; // number of filter outputs

    ///////////////////////////////////////////////////

    // filter model
    size_t inputs_{1}; // number of filter inputs

    matrix_wrapper<float> A_{}; // state transition matrix (states_ x inputs_)

    matrix_wrapper<float> B_{}; // input transition matrix (states_ x inputs_)

    matrix_wrapper<float> u_{}; // input vector (inputs_ x 1)

    matrix_wrapper<float> P_{}; // state covariance matrix (states_ x states_)

    matrix_wrapper<float> Q_{}; // input covariance matrix (inputs_ x inputs_)

    matrix_wrapper<float> x_{}; // state vector (states_ x 1)

    matrix_wrapper<float> xP_{}; // predicted state vector (states_ x 1)

    ///////////////////////////////////////////////////

    // measurement model
    size_t measurements_{1}; // number of measerements performed

    matrix_wrapper<float> H_{}; // measurement transformation matrix (measurements_ x states_)

    matrix_wrapper<float> z_{}; // measurement vector (measurements_ x  1)

    matrix_wrapper<float> R_{}; // process noise (measurement uncertainty)
                                // (measurements_ x measurements_)

    matrix_wrapper<float> y_{}; // innovation vector (measurements_ x 1)

    matrix_wrapper<float> S_{}; // residual covariance (measurements_ x measurements_)

    matrix_wrapper<float> K_{}; // kalman gain (states_ x measurements_)

    ///////////////////////////////////////////////////

    // temporary objects for move semantics (operators taking rvalue refs)
    matrix_wrapper<float> tempBQ_{};

    matrix_wrapper<float> tempAP_{};

    matrix_wrapper<float> tempHx_{};

    matrix_wrapper<float> tempHP_{};

    matrix_wrapper<float> tempH_{};

    matrix_wrapper<float> tempPHt_{};

    matrix_wrapper<float> tempS_{};

    matrix_wrapper<float> tempKy_{};
};

#endif // KALMAN_HPP