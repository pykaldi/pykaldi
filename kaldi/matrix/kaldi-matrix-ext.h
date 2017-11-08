#ifndef PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_
#define PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_ 1

#include "matrix/kaldi-matrix.h"

/// Shims for kaldi::Vector<Real> methods that we cannot wrap in
/// in matrix/kaldi-matrix.clif

namespace kaldi {

template<typename Real, typename OtherReal>
void CopyFromMat(MatrixBase<Real> *self, const MatrixBase<OtherReal> & M, MatrixTransposeType trans = kNoTrans) {
  self->CopyFromMat(M);
}

template<typename Real>
void CopyFromCompressed(MatrixBase<Real> *self, const CompressedMatrix &M) {
  self->CopyFromMat(M);
}

template<typename Real, typename OtherReal>
void CopyFromSp(MatrixBase<Real> *self, const SpMatrix<OtherReal> &M) {
  self->CopyFromSp(M);
}

template<typename Real, typename OtherReal>
void CopyFromTp(MatrixBase<Real> *self, const TpMatrix<OtherReal> &M, MatrixTransposeType trans = kNoTrans) {
  self->CopyFromTp(M);
}

template<typename Real, typename OtherReal>
void AddSp(MatrixBase<Real> *self, const Real alpha, const SpMatrix<OtherReal> &S) {
  self->AddSp(alpha, S);
}

template<typename Real>
void AddSpMat(MatrixBase<Real> *self, const Real alpha, const SpMatrix<Real> &A, const MatrixBase<Real> &B, MatrixTransposeType transB, const Real beta)  {
  self->AddSpMat(alpha, A, B, transB, beta);
}

template<typename Real>
void AddTpMat(MatrixBase<Real> *self, const Real alpha, const TpMatrix<Real> &A, MatrixTransposeType transA, const MatrixBase<Real> &B, MatrixTransposeType transB, const Real beta) {
  self->AddTpMat(alpha, A, transA, B, transB, beta);
}

template<typename Real>
void AddMatSp(MatrixBase<Real> *self, const Real alpha, const MatrixBase<Real> &A, MatrixTransposeType transA, const SpMatrix<Real> &B, const Real beta) {
  self->AddMatSp(alpha, A, transA, B, beta);
}

template<typename Real>
void AddMatTp(MatrixBase<Real> *self, const Real alpha, const MatrixBase<Real> &A, MatrixTransposeType transA, const TpMatrix<Real> &B, MatrixTransposeType transB, const Real beta) {
  self->AddMatTp(alpha, A, transA, B, transB, beta);
}

template<typename Real>
void AddTpTp(MatrixBase<Real> *self, const Real alpha, const TpMatrix<Real> &A, MatrixTransposeType transA, TpMatrix<Real> &B, MatrixTransposeType transB, const Real beta) {
  self->AddTpTp(alpha, A, transA, B, transB, beta);
}

template<typename Real>
void AddSpSp(MatrixBase<Real> *self, const Real alpha, const SpMatrix<Real> &A, const SpMatrix<Real> &B, const Real beta) {
  self->AddSpSp(alpha, A, B, beta);
}

template<typename Real>
void Invert(MatrixBase<Real> *self, Real *log_det = NULL, Real *det_sign = NULL) {
  self->Invert(log_det, det_sign, true);
}

template<typename Real>
void InvertDouble(MatrixBase<Real> *self, Real *log_det = NULL, Real *det_sign = NULL) {
  self->InvertDouble(log_det, det_sign, true);
}

template<typename Real>
void CopyCols(MatrixBase<Real> *self, const MatrixBase<Real> &src, const std::vector<MatrixIndexT> &indices) {
  self->CopyCols(src, indices.data());
}

template<typename Real>
void CopyRows(MatrixBase<Real> *self, const MatrixBase<Real> &src, const std::vector<MatrixIndexT> &indices) {
  self->CopyRows(src, indices.data());
}

template<typename Real>
void AddCols(MatrixBase<Real> *self, const MatrixBase<Real> &src, const std::vector<MatrixIndexT> &indices) {
  self->AddCols(src, indices.data());
}

template<typename Real>
void AddRows(MatrixBase<Real> *self, Real alpha, const MatrixBase<Real> &src, const std::vector<MatrixIndexT> &indices) {
  self->AddRows(alpha, src, indices.data());
}

}

#endif //PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_
