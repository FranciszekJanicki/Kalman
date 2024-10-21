#ifndef KALMAN_HPP
#define KALMAN_HPP

#include "matrix.hpp"
#include <stdfloat>

using Matrix = Linalg::Matrix<std::float64_t>;

struct FilterModel {
    std::size_t states{}; // number of filter units
    std::size_t inputs{}; // number of filter inputs

    Matrix A{}; // state transition matrix (numStates x numInputs)
    Matrix x{}; // state vector_data (numStates x 1)
    Matrix B{}; // input transition matrix (numStates x numInputs)
    Matrix u{}; // input vector_data (numInputs x 1)
    Matrix P{}; // state covariance matrix (numStates x numStates)
    Matrix Q{}; // input covariance matrix (numInputs x numInputs)
};

struct MeasureModel {
    std::size_t states{};       // number of filter units
    std::size_t measurements{}; // number of meauserements performed

    Matrix H{}; // measurement transformation matrix (measurements x states)
    Matrix z{}; // measurement vector_data (numMeasures x  1)
    Matrix R{}; // process noise (measurement uncertainty) (measurements x measurements)
    Matrix y{}; // innovation (measurements x 1)
    Matrix S{}; // residual covariance (measurements x measurements)
    Matrix K{}; // Kalman gain (states x measurements)
};

class Kalman {
public:
    Kalman(const FilterModel& filter, const MeasureModel& measure);
    Kalman(FilterModel&& filter, MeasureModel&& measure) noexcept;

    [[nodiscard]] Matrix&& state() && noexcept;
    [[nodiscard]] const Matrix& state() const& noexcept;

    void inputs(const Matrix& inputs);
    void inputs(Matrix&& inputs) noexcept;

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
    Matrix A_{};  // state transition matrix (states_ x inputs_)
    Matrix B_{};  // input transition matrix (states_ x inputs_)
    Matrix u_{};  // input vector_data (inputs_ x 1)
    Matrix P_{};  // state covariance matrix (states_ x states_)
    Matrix Q_{};  // input covariance matrix (inputs_ x inputs_)
    Matrix x_{};  // state vector_data (states_ x 1)
    Matrix xP_{}; // predicted state vector_data (states_ x 1)

    ///////////////////////////////////////////////////

    Matrix H_{}; // measurement transformation matrix (measurements_ x states_)
    Matrix z_{}; // measurement vector_data (measurements_ x  1)
    Matrix R_{}; // process noise (measurement uncertainty) (measurements_ x measurements_)
    Matrix y_{}; // innovation vector_data (measurements_ x 1)
    Matrix S_{}; // residual covariance (measurements_ x measurements_)
    Matrix K_{}; // Kalman gain (states_ x measurements_)

    ///////////////////////////////////////////////////

    // temporary objects for move semantics (operators taking rvalue refs)
    Matrix tempHP_{};
    Matrix tempH_{};
    Matrix tempS_{};
};

#endif // KALMAN_HPP