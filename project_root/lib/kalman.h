#ifndef KALMAN_H
#define KALMAN_H

#include "matrix.hpp"
// #include "Eigen/Dense"

struct FilterModel {
    std::size_t        states{}; // number of filter units
    std::size_t        inputs{}; // number of filter inputs
    mtx::Matrix<float> A{};      // state transition matrix (numStates x numInputs)
    mtx::Matrix<float> x{};      // state vector (numStates x 1)
    mtx::Matrix<float> B{};      // input transition matrix (numStates x numInputs)
    mtx::Matrix<float> u{};      // input vector (numInputs x 1)
    mtx::Matrix<float> P{};      // state covariance matrix (numStates x numStates)
    mtx::Matrix<float> Q{};      // input covariance matrix (numInputs x numInputs)
};

struct MeasureModel {
    std::size_t        states{};       // number of filter units
    std::size_t        measurements{}; // number of meauserements performed
    mtx::Matrix<float> H{};            // measurement transformation matrix (measurements x states)
    mtx::Matrix<float> z{};            // measurement vector (numMeasures x  1)
    mtx::Matrix<float> R{};            // process noise (measurement uncertainty)
                                       // (measurements x measurements)
    mtx::Matrix<float> y{};            // innovation (measurements x 1)
    mtx::Matrix<float> S{};            // residual covariance (measurements x measurements)
    mtx::Matrix<float> K{};            // kalman gain (states x measurements)
};

class Kalman {
public:
    explicit Kalman() = default;
    explicit Kalman(FilterModel& filter, MeasureModel& measure);
    explicit Kalman(FilterModel&& filter, MeasureModel&& measure);

    [[nodiscard]] inline const mtx::Matrix<float>& getState() const noexcept;
    [[nodiscard]] inline double                    getTime() const noexcept;
    [[nodiscard]] inline bool                      getIsInitialized() const noexcept;
    inline void                                    setInputs(const mtx::Matrix<float>& inputs) noexcept;

private:
    void predict() noexcept;
    void update() noexcept;

    bool  isInitialized_{false};
    float stepTime_{1.0f};
    float currentTime_{0.0f};
    float startTime_{0.0f};

    // common model
    size_t states_{1}; // number of filter outputs

    // filter model
    size_t             inputs_{1};           // number of filter inputs
    mtx::Matrix<float> A_{states_, inputs_}; // state transition matrix (states_ x inputs_)
    mtx::Matrix<float> B_{states_, inputs_}; // input transition matrix (states_ x inputs_)
    mtx::Matrix<float> u_{inputs_, 1};       // input vector (inputs_ x 1)
    mtx::Matrix<float> P_{states_, states_}; // state covariance matrix (states_ x states_)
    mtx::Matrix<float> Q_{inputs_, inputs_}; // input covariance matrix (inputs_ x inputs_)
    mtx::Matrix<float> x_{states_, 1};       // state vector (states_ x 1)
    mtx::Matrix<float> xP_{states_, 1};      // predicted state vector (states_ x 1)

    // measurement model
    size_t             measurements_{1};                 // number of measerements performed
    mtx::Matrix<float> H_{measurements_, states_};       // measurement transformation matrix (measurements_ x states_)
    mtx::Matrix<float> z_{measurements_, 1};             // measurement vector (measurements_ x  1)
    mtx::Matrix<float> R_{measurements_, measurements_}; // process noise (measurement uncertainty)
                                                         // (measurements_ x measurements_)
    mtx::Matrix<float> y_{measurements_, 1};             // innovation vector (measurements_ x 1)
    mtx::Matrix<float> S_{measurements_, measurements_}; // residual covariance (measurements_ x measurements_)
    mtx::Matrix<float> K_{states_, measurements_};       // kalman gain (states_ x measurements_)

    // temporary objects for move semantics (operators taking lvalue refs)
    mtx::Matrix<float> tempBQ_{};
    mtx::Matrix<float> tempAP_{};
    mtx::Matrix<float> tempHx_{};
    mtx::Matrix<float> tempHP_{};
    mtx::Matrix<float> tempH_{};
    mtx::Matrix<float> tempPHt_{};
    mtx::Matrix<float> tempS_{};
    mtx::Matrix<float> tempKy_{};
};

#endif // KALMAN_H