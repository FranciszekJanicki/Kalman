#include "kalman.h"
#include <cassert>
#include <utility>

// if passed by non-const lvalue reference, the direct parameter initialization would be impossible, as cant initialize non-const lvalue ref
// with rvalue

// if passed by const lvalue reference, direct parameter initialization would be possible (const reference can reference both r and l
// values), but since const reference is a reference to const object, the const object has all members const, which would explicitly forbid
// moves to members, like seen below

// passed by value to allow both rvalues (move parameter initialization) and lvalues (copy parameter initialization) and direct parameter
// initialization without forbidding moves
Kalman::Kalman(FilterModel& filter, MeasureModel& measure) :
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

// /* using move semantics to avoid copying (apart from when explicitly need to copy to not 'steal' from rhs)
//    Using temporary variables to not run void methods on the const objects (its forbidden either way)
// */
// void predict() {
//     // using temp variables to store result of one operation and use it to perform next

//     /* predict state        */
//     // xP = A*x

//     xP_ = std::move(A_ * x_);
//     // assume predicted state
//     x_ = std::move(xP_);

//     /************************/

//     /* predict covariance   */
//     // P = A*P*A + B*Q*B

//     // P = A*P*A
//     tempAP_ = std::move(A_ * P_); // tempPA = P*A
//     P_ = std::move(tempAP_ * A_); // P = tempAP*A

//     // P += B*Q*B
//     tempBQ_ = std::move(B_ * Q_) // tempBP = B*Q
//     P_ += std::move(tempBQ_ * B); // P += tempBQ*Q
// }

// void Kalman::update() {
//     /* calculate innovation */
//     // y = z - H*x

//     tempHx_ = std::move(H_ * x_); // tempHx = H*x
//     y_ = std::move(z_ - tempHx_); // y = z - tempHx

//     /************************/

//     /* calculate residual covariance */
//     // S = H*P*H' + R

//     tempHP_ = std::move(H_ * P_); // tempHP = H*P
//     tempH_ = H_; // tempH = H (to not run .transpose on H)
//     tempH_.transpose();
//     S_ = std::move(tempHP_ * tempH); // S = tempHP*tempH'
//     S_ += R_; // S += R

//     /*************************/

//     /* calculate kalman gain */
//     // K = P*H' * S^-1

//     tempPHt_ = std::move(P_ * tempH_.transpose()); // tempPH' = P*H' (temphH was transposed above)
//     tempS_ = S_; // tempS = S
//     tempS_.invert();  tempS = S^-1;
//     K_ = std::move(tempPHt_ * tempS); // K = tempPHt * S^-1

//     /**************************/

//     /* correct state prediction */
//     // x = x * K*y

//     tempKy_ = std::move(K_ * y_);
//     x_ *= tempKy_;

//     /* now, the x_ is current filter output! */

//     /***************************/

//     /* correct state covariance */
//     // P = P - K*(H*P)

//     tempHP_ = std::move(H_ * P_);
//     P_ -= std::move(K_ * tempHP_);
// }

/* +=, -=, *=, /=, etc, operators are better, as they perform operations on the reference
to lhs; +, -, *, /, etc operators perform operation on some local object, return its copy and then when
assigning that result somewhere, calling assingment operator between lhs and this result copy */

/*
x = x + y; called operator+ on x and y, returned copy of result, then called assingment operator on lhs and rhs (result)
x += y; called += operator directly on lhs and rhs
*/

// using std::move to explicitly cast to rvalue and force move assingment operator, as even though operators (like functions) return
// temporaries, which are rvalues, you need explicit cast to perform move!
void Kalman::predict() {
    assert(isInitialized_ == true);

    // using temp variables to store result of one operation and use it to perform next

    /* predict state        */
    // xP = A*x

    xP_ = std::move(A_ * x_);

    /************************/

    /* predict covariance   */
    // P = A*P*A + B*Q*B

    // P = A*P*A
    P_ = std::move(A_ * P_ * A_);

    // P += B*Q*B
    P_ += std::move(B_ * Q_ * B_);
}

void Kalman::update() {
    assert(isInitialized_ == true);

    /* calculate innovation */
    // y = z - H*x

    y_ = std::move(z_ - H_ * x_);

    /************************/

    /* calculate residual covariance */
    // S = H*P*H' + R

    tempH_ = H_;
    tempH_.transpose();
    S_ = std::move(H_ * P_ * tempH_ + R_);

    /*************************/

    /* calculate kalman gain */
    // K = P*H' * S^-1

    tempS_ = S_;
    tempS_.invert();
    K_ = std::move(P_ * tempH_ * tempS_); // tempH is already transposed above

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