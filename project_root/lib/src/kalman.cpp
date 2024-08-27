#include "kalman.h"
#include <cassert>
#include <utility>

Kalman::Kalman(FilterModel& filter, MeasureModel& measure) :
    A_{std::move(filter.A_)}, B_{std::move(filter.B_)}, P_{std::move(filter.P_)}, Q_{std::move(filter.Q_)}, x_{std::move(filter.x_)},
    u_{std::move(filter.u_)}, H_{std::move(measure.H_)}, R_{std::move(measure.R_)}, S_{std::move(measure.S_)}, K_{std::move(measure.K_)},
    z_{std::move(measure.z_)}, y_{std::move(measure.z_)}, inputs_{filter.inputs_}, states_{filter.inputs_}, measurements_{
                                                                                                                measure.measurements_} {
    assert(filter.states_ == measure.states_);
    isInitialized_ = true;
}

Kalman::Kalman(FilterModel filter, MeasureModel measure) :
    A_{std::move(filter.A_)}, B_{std::move(filter.B_)}, P_{std::move(filter.P_)}, Q_{std::move(filter.Q_)}, x_{std::move(filter.x_)},
    u_{std::move(filter.u_)}, H_{std::move(measure.H_)}, R_{std::move(measure.R_)}, S_{std::move(measure.S_)}, K_{std::move(measure.K_)},
    z_{std::move(measure.z_)}, y_{std::move(measure.z_)}, inputs_{filter.inputs_}, states_{filter.inputs_}, measurements_{
                                                                                                                measure.measurements_} {
    assert(filter.states_ == measure.states_);
    isInitialized_ = true;
}

bool Kalman::getIsInitialized() const {
    return isInitialized_;
}

const mtx::Matrix<float>& Kalman::getState() const {
    return x_;
}

double Kalman::getTime() const {
    return currentTime_;
}

void Kalman::setInputs(const mtx::Matrix<float>& inputs) {
    u_ = inputs;
}

// using std::move to explicitly cast to rvalue and force move assingment operator, as even though operators (like functions) return
// temporaries, which are rvalues, you need explicit cast to perform move!
void Kalman::predict() {
    if (!isInitialized_) {
        LOG("Filter unitialized!");
        return;
    }

    // using temp variables to store result of one operation and use it to perform next

    /* predict state        */
    // xP = A*x

    xP_ = std::move(A_ * x_);

    /************************/

    /* predict covariance   */
    // P = A*P*A + B*Q*B

    // P = A*P*A
    P_ = std::move((A_ * P_) * A_);

    // P += B*Q*B
    P_ += std::move((B_ * Q_) * B_);
}

void Kalman::update() {
    if (!isInitialized_) {
        LOG("Filter unitialized!");
        return;
    }

    /* calculate innovation */
    // y = z - H*x

    y_ = std::move(z_ - (H_ * x_));

    /************************/

    /* calculate residual covariance */
    // S = H*P*H' + R

    tempH_ = H_;
    tempH_.transpose();
    S_ = std::move((H_ * P_ * tempH_) + R_);

    /*************************/

    /* calculate kalman gain */
    // K = P*H' * S^-1

    tempS_ = S_;
    tempS_.invert();
    K_ = std::move((P_ * tempH_) * tempS_); // tempH is already transposed above

    /**************************/

    /* correct state prediction */
    // x = x * K*y

    x_ *= std::move(K_ * y_);

    /* now, the x_ is current filter output! */

    /***************************/

    /* correct state covariance */
    // P = P - K*(H*P)

    tempHP_ = std::move(H_ * P_);
    P_ -= std::move(K_ * tempHP_); // tempHP as operators run as compiler sees them (matrices mult order matters!!!)
}