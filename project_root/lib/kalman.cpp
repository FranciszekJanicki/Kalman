#include "kalman.h"
#include <cassert>
#include <iostream>
#include <string_view>
#include <utility>

inline static void LOG(std::string_view error) {
    std::cerr << error;
}

Kalman::Kalman(FilterModel& filter, MeasureModel& measure) :
    A_{filter.A}, B_{filter.B}, P_{filter.P}, Q_{filter.Q}, x_{filter.x}, u_{filter.u}, H_{measure.H}, R_{measure.R},
    S_{measure.S}, K_{measure.K}, z_{measure.z}, y_{measure.z}, inputs_{filter.inputs}, states_{filter.inputs},
    measurements_{measure.measurements} {
    assert(filter.states == measure.states);
    isInitialized_ = true;
}

Kalman::Kalman(FilterModel&& filter, MeasureModel&& measure) :
    A_{std::move(filter.A)}, B_{std::move(filter.B)}, P_{std::move(filter.P)}, Q_{std::move(filter.Q)},
    x_{std::move(filter.x)}, u_{std::move(filter.u)}, H_{std::move(measure.H)}, R_{std::move(measure.R)},
    S_{std::move(measure.S)}, K_{std::move(measure.K)}, z_{std::move(measure.z)}, y_{std::move(measure.z)},
    inputs_{filter.inputs}, states_{filter.inputs}, measurements_{measure.measurements} {
    assert(filter.states == measure.states);
    isInitialized_ = true;
}

bool Kalman::getIsInitialized() const noexcept {
    return isInitialized_;
}

const mtx::Matrix<float>& Kalman::getState() const noexcept {
    return x_;
}

double Kalman::getTime() const noexcept {
    return currentTime_;
}

void Kalman::setInputs(const mtx::Matrix<float>& inputs) noexcept {
    u_ = inputs;
}

void Kalman::predict() noexcept {
    if (!isInitialized_) {
        LOG("Filter unitialized!");
        return;
    }

    // using temp variables to store result of one operation and use it to perform next

    /* predict state        */
    // xP = A*x

    xP_ = A_ * x_;

    /************************/

    /* predict covariance   */
    // P = A*P*A + B*Q*B

    // P = A*P*A
    P_ = ((A_ * P_) * A_);

    // P += B*Q*B
    P_ += ((B_ * Q_) * B_);
}

void Kalman::update() noexcept {
    if (!isInitialized_) {
        LOG("Filter unitialized!");
        return;
    }

    /* calculate innovation */
    // y = z - H*x

    y_ = z_ - (H_ * x_);

    /************************/

    /* calculate residual covariance */
    // S = H*P*H' + R

    tempH_ = H_;
    tempH_.transpose();
    S_ = (H_ * P_ * tempH_) + R_;

    /*************************/

    /* calculate kalman gain */
    // K = P*H' * S^-1

    tempS_ = S_;
    tempS_.invert();
    K_ = ((P_ * tempH_) * tempS_); // tempH is already transposed above

    /**************************/

    /* correct state prediction */
    // x = x * K*y

    x_ *= K_ * y_;

    /* now, the x_ is current filter output! */

    /***************************/

    /* correct state covariance */
    // P = P - K*(H*P)

    tempHP_ = H_ * P_;
    P_ -= K_ * tempHP_; // tempHP as operators run as compiler sees them (matrices mult order matters!!!)
}