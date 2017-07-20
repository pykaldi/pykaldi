#ifndef PYKALDI_MATRIX_KALDI_VECTOR_EXT_H_
#define PYKALDI_MATRIX_KALDI_VECTOR_EXT_H_ 1

#include "matrix/kaldi-vector.h"

/// Proxy functions for kaldi::Vector<float> methods that we cannot wrap in
/// matrix/kaldi-vector.clif since they refer to Matrix types.

namespace kaldi {

void AddMatVec(VectorBase<float> *self, const float alpha,
               const MatrixBase<float> &M, const MatrixTransposeType trans,
               const VectorBase<float> &v, const float beta) {
  self->AddMatVec(alpha, M, trans, v, beta);
}

void AddMatSvec(VectorBase<float> *self, const float alpha,
                const MatrixBase<float> &M, const MatrixTransposeType trans,
                const VectorBase<float> &v, const float beta) {
  self->AddMatSvec(alpha, M, trans, v, beta);
}

void AddSpVec(VectorBase<float> *self, const float alpha,
              const SpMatrix<float> &M,
              const VectorBase<float> &v, const float beta) {
  self->AddSpVec(alpha, M, v, beta);
}

void AddTpVec(VectorBase<float> *self, const float alpha,
              const TpMatrix<float> &M,
              const MatrixTransposeType trans,
              const VectorBase<float> &v, const float beta) {
  self->AddTpVec(alpha, M, trans, v, beta);
}

void MulTp(VectorBase<float> *self, const TpMatrix<float> &M,
           const MatrixTransposeType trans) {
  self->MulTp(M, trans);
}

void Solve(VectorBase<float> *self, const TpMatrix<float> &M,
           const MatrixTransposeType trans) {
  self->Solve(M, trans);
}

void CopyRowsFromMat(VectorBase<float> *self, const MatrixBase<float> &M) {
  self->CopyRowsFromMat(M);
}

void CopyColsFromMat(VectorBase<float> *self, const MatrixBase<float> &M) {
  self->CopyColsFromMat(M);
}

void CopyRowFromMat(VectorBase<float> *self, const MatrixBase<float> &M,
                    MatrixIndexT row) {
  self->CopyRowFromMat(M, row);
}

void CopyColFromMat(VectorBase<float> *self, const MatrixBase<float> &M,
                    MatrixIndexT col) {
  self->CopyColFromMat(M, col);
}

void CopyDiagFromMat(VectorBase<float> *self, const MatrixBase<float> &M) {
  self->CopyDiagFromMat(M);
}

void CopyFromPacked(VectorBase<float> *self, const PackedMatrix<float> &M) {
  self->CopyFromPacked(M);
}

void CopyDiagFromPacked(VectorBase<float> *self, const PackedMatrix<float> &M) {
  self->CopyDiagFromPacked(M);
}

inline void CopyDiagFromSp(VectorBase<float> *self, const SpMatrix<float> &M) {
  self->CopyDiagFromPacked(M);
}

inline void CopyDiagFromTp(VectorBase<float> *self, const TpMatrix<float> &M) {
  self->CopyDiagFromPacked(M);
}

void AddRowSumMat(VectorBase<float> *self, float alpha,
                  const MatrixBase<float> &M, float beta = 1.0) {
  self->AddRowSumMat(alpha, M, beta);
}

void AddColSumMat(VectorBase<float> *self, float alpha,
                  const MatrixBase<float> &M, float beta = 1.0) {
  self->AddColSumMat(alpha, M, beta);
}

void AddDiagMat2(VectorBase<float> *self, float alpha,
                 const MatrixBase<float> &M,
                 MatrixTransposeType trans = kNoTrans, float beta = 1.0) {
  self->AddDiagMat2(alpha, M, trans, beta);
}

void AddDiagMatMat(VectorBase<float> *self, float alpha,
                   const MatrixBase<float> &M, MatrixTransposeType transM,
                   const MatrixBase<float> &N, MatrixTransposeType transN,
                   float beta = 1.0) {
  self->AddDiagMatMat(alpha, M, transM, N, transN, beta);
}

template<typename Real>
Real VecMatVec(const VectorBase<Real> &v1, const MatrixBase<Real> &M,
               const VectorBase<Real> &v2);

}  // namespace kaldi

#endif  // KALDI_MATRIX_KALDI_VECTOR_EXT_H_
