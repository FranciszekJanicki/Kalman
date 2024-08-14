#include "kalman.h"


Kalman::Kalman(int numStates, int numInputs, float startTime, float stepTime) : numStates_ {numStates},
                                                                                numInputs_ {numInputs}, 
                                                                                startTime_ {startTime},
                                                                                stepTime_ {stepTime} {
    assert(numStates > 0);
    assert(numInputs > 0);
    assert(startTime > 0.0f);
    assert(stepTime > 0.0f);
}

const mtx::Matrix &Kalman::getState() const {

}

float Kalman::getTime() const {

}

void Kalman::init(const KalmanMatrixes &models) {
    cMatrixes_[MatrixID::kMatrixA] = models.dynamics_;
    cMatrixes_[MatrixID::kMatrixB] = models.estimateError_;
    cMatrixes_[MatrixID::kMatrixP] = models.measurementNoise_;
    cMatrixes_[MatrixID::kMatrixQ] = models.output_;
    cMatrixes_[MatrixID::kVectorX] = models.processNoise_;
    cMatrixes_[MatrixID::kVectorU] = models.initEstimateError_;

    for (const auto &[key, value] : cMatrixes_) {
        if (value.getRows() == 0 || value.getCols() == 0) {
            throw {
                std::runtime_error("Incorrect dimensions");
            }
        }
    }

    numStates_ = cMatrixes[MatrixID::kMatrixA].getRows();
    numInputs_ = cMatrixes[MatrixID::kVectorU].getRows();

}
void Kalman::update(const mtx::Matrix &measured, float stepTime, const mtx::Matrix &dynamics) {

}
                