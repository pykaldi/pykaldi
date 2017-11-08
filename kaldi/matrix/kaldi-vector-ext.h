#ifndef PYKALDI_MATRIX_KALDI_VECTOR_EXT_H_
#define PYKALDI_MATRIX_KALDI_VECTOR_EXT_H_ 1

#include "matrix/kaldi-vector.h"

/// Shims for kaldi::Vector<Real> methods that we cannot wrap in
/// matrix/kaldi-vector.clif.

namespace kaldi {

template<typename Real, typename OtherReal>
void CopyFromPacked(VectorBase<Real> *self, const PackedMatrix<OtherReal> &M) {
  self->CopyFromPacked(M);
}

template<typename Real, typename OtherReal>
void CopyFromVec(VectorBase<Real> *self, const VectorBase<OtherReal> &v) {
  self->CopyFromVec(v);
}

template<typename Real, typename OtherReal>
void AddVec(VectorBase<Real> *self, const Real alpha, const VectorBase<OtherReal> &v) {
  self->AddVec(alpha, v);
}

template<typename Real, typename OtherReal>
void AddVec2(VectorBase<Real> *self, const Real alpha, const VectorBase<OtherReal> &v) {
  self->AddVec2(alpha, v);
}

template<typename Real>
void AddMatVec(VectorBase<Real> *self, const Real alpha,
               const MatrixBase<Real> &M, const MatrixTransposeType trans,
               const VectorBase<Real> &v, const Real beta) {
  self->AddMatVec(alpha, M, trans, v, beta);
}

template<typename Real>
void AddMatSvec(VectorBase<Real> *self, const Real alpha,
                const MatrixBase<Real> &M, const MatrixTransposeType trans,
                const VectorBase<Real> &v, const Real beta) {
  self->AddMatSvec(alpha, M, trans, v, beta);
}

template<typename Real>
void AddSpVec(VectorBase<Real> *self, const Real alpha,
              const SpMatrix<Real> &M,
              const VectorBase<Real> &v, const Real beta) {
  self->AddSpVec(alpha, M, v, beta);
}

template<typename Real>
void AddTpVec(VectorBase<Real> *self, const Real alpha,
              const TpMatrix<Real> &M,
              const MatrixTransposeType trans,
              const VectorBase<Real> &v, const Real beta) {
  self->AddTpVec(alpha, M, trans, v, beta);
}

template<typename Real, typename OtherReal>
void MulElements(VectorBase<Real> *self, const VectorBase<OtherReal> &v) {
  self->MulElements(v);
}

template<typename Real, typename OtherReal>
void DivElements(VectorBase<Real> *self, const VectorBase<OtherReal> &v) {
  self->DivElements(v);
}

template<typename Real>
void MulTp(VectorBase<Real> *self, const TpMatrix<Real> &M,
           const MatrixTransposeType trans) {
  self->MulTp(M, trans);
}

template<typename Real>
void Solve(VectorBase<Real> *self, const TpMatrix<Real> &M,
           const MatrixTransposeType trans) {
  self->Solve(M, trans);
}

template<typename Real, typename OtherReal>
void CopyRowsFromMat(VectorBase<Real> *self, const MatrixBase<OtherReal> &M) {
  self->CopyRowsFromMat(M);
}

template<typename Real>
void CopyColsFromMat(VectorBase<Real> *self, const MatrixBase<Real> &M) {
  self->CopyColsFromMat(M);
}

template<typename Real, typename OtherReal>
void CopyRowFromMat(VectorBase<Real> *self, const MatrixBase<OtherReal> &M,
                    MatrixIndexT row) {
  self->CopyRowFromMat(M, row);
}

template<typename Real, typename OtherReal>
void CopyRowFromSp(VectorBase<Real> *self, const SpMatrix<OtherReal> &S,
                   MatrixIndexT row) {
  self->CopyRowFromSp(S, row);
}

template<typename Real, typename OtherReal>
void CopyColFromMat(VectorBase<Real> *self, const MatrixBase<OtherReal> &M,
                    MatrixIndexT col) {
  self->CopyColFromMat(M, col);
}

template<typename Real>
void CopyDiagFromMat(VectorBase<Real> *self, const MatrixBase<Real> &M) {
  self->CopyDiagFromMat(M);
}

template<typename Real>
void CopyDiagFromPacked(VectorBase<Real> *self, const PackedMatrix<Real> &M) {
  self->CopyDiagFromPacked(M);
}

template<typename Real>
inline void CopyDiagFromSp(VectorBase<Real> *self, const SpMatrix<Real> &M) {
  self->CopyDiagFromPacked(M);
}

template<typename Real>
inline void CopyDiagFromTp(VectorBase<Real> *self, const TpMatrix<Real> &M) {
  self->CopyDiagFromPacked(M);
}

template<typename Real>
void AddRowSumMat(VectorBase<Real> *self, Real alpha,
                  const MatrixBase<Real> &M, Real beta = 1.0) {
  self->AddRowSumMat(alpha, M, beta);
}

template<typename Real>
void AddColSumMat(VectorBase<Real> *self, Real alpha,
                  const MatrixBase<Real> &M, Real beta = 1.0) {
  self->AddColSumMat(alpha, M, beta);
}

template<typename Real>
void AddDiagMat2(VectorBase<Real> *self, Real alpha,
                 const MatrixBase<Real> &M,
                 MatrixTransposeType trans = kNoTrans, Real beta = 1.0) {
  self->AddDiagMat2(alpha, M, trans, beta);
}

template<typename Real>
void AddDiagMatMat(VectorBase<Real> *self, Real alpha,
                   const MatrixBase<Real> &M, MatrixTransposeType transM,
                   const MatrixBase<Real> &N, MatrixTransposeType transN,
                   Real beta = 1.0) {
  self->AddDiagMatMat(alpha, M, transM, N, transN, beta);
}

template<typename Real>
Real VecMatVec(const VectorBase<Real> &v1, const MatrixBase<Real> &M,
               const VectorBase<Real> &v2);

}  // namespace kaldi

#endif  // PYKALDI_MATRIX_KALDI_VECTOR_EXT_H_
