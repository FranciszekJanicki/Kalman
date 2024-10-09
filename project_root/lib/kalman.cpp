#include "kalman.hpp"
#include <cstdio>
#include <utility>

kalman::kalman(const filter_model& filter, const measure_model& measure) :

    states_{filter.inputs},
    inputs_{filter.inputs},
    A_(filter.A),
    B_(filter.B),
    u_(filter.u),
    P_(filter.P),
    Q_(filter.Q),
    x_(filter.x),
    measurements_{measure.measurements},
    H_(measure.H),
    z_(measure.z),
    R_(measure.R),
    y_(measure.z),
    S_(measure.S),
    K_(measure.K)
{
    puts("Initialized");
    assert(filter.states == measure.states);
    is_initialized_ = true;
}

kalman::kalman(filter_model&& filter, measure_model&& measure) noexcept :
    states_{filter.inputs},
    inputs_{filter.inputs},
    A_(std::move(filter.A)),
    B_(std::move(filter.B)),
    u_(std::move(filter.u)),
    P_(std::move(filter.P)),
    Q_(std::move(filter.Q)),
    x_(std::move(filter.x)),
    measurements_{measure.measurements},
    H_(std::move(measure.H)),
    z_(std::move(measure.z)),
    R_(std::move(measure.R)),
    y_(std::move(measure.z)),
    S_(std::move(measure.S)),
    K_(std::move(measure.K))
{
    puts("Initialized");
    assert(filter.states == measure.states);
    is_initialized_ = true;
}

matrix_wrapper<float>&& kalman::state() && noexcept
{
    return std::move(x_);
}

const matrix_wrapper<float>& kalman::state() const& noexcept
{
    return x_;
}

void kalman::inputs(const matrix_wrapper<float>& inputs)
{
    u_ = inputs;
}

void kalman::inputs(matrix_wrapper<float>&& inputs) noexcept
{
    u_ = std::forward<matrix_wrapper<float>>(inputs);
}

void kalman::predict()
{
    if (!is_initialized_) {
        puts("Filter unitialized!");
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

void kalman::update()
{
    if (!is_initialized_) {
        puts("Filter unitialized!");
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

void kalman::print_state() const noexcept
{
    x_.print();
}

void kalman::print_predicted() const noexcept
{
    xP_.print();
}