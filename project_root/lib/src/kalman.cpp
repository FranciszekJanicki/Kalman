#include "kalman.h"
#include <cassert>
#include <utility>


Kalman::Kalman(const FilterModel &filter, const MeasureModel &measure) :  A_ {std::move(filter.A_)}, B_ {std::move(filter.B_)},
                                                                          P_ {std::move(filter.P_)}, Q_ {std::move(filter.Q_)},
                                                                          x_ {std::move(filter.x_)}, u_ {std::move(filter.u_)},
                                                                          H_ {std::move(measure.H_)}, R_ {std::move(measure.R_)},
                                                                          S_ {std::move(measure.S_)}, K_ {std::move(measure.K_)},
                                                                          z_ {std::move(measure.z_)}, y_ {std::move(measure.z_)},
                                                                          inputs_ {filter.inputs_}, states_ {filter.inputs_},
                                                                          measurements_ {measure.measurements_} {
    assert(filter.states_ == measure.states_);
    isInitialized_ = true;
}

bool Kalman::getIsInitialized() const {
    return isInitialized_;
}

const mtx::Matrix<float> &Kalman::getState() const {
    return x_;
}

double Kalman::getTime() const {
    return currentTime_;
}

void Kalman::setInputs(const mtx::Matrix<float> &inputs) {
    u_ = inputs;
}

void Kalman::update() {

}

void Kalman::predict() {

}
       