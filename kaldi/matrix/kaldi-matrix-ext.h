#ifndef PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_
#define PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_ 1

#include "matrix/kaldi-matrix.h"

/// Proxy functions for kaldi::Matrix<float> methods that we cannot wrap
/// in matrix/kaldi-matrix.clif

namespace kaldi {

	/**
	* Fix methods refering to other types
	**/
	void CopyFromMat(MatrixBase<float> *self, CompressedMatrix &M){
		self->CopyFromMat(M);
	}

	void CopyFromSp(MatrixBase<float> *self, SpMatrix<float> &M){
		self->CopyFromSp(M);
	}

	void CopyFromTp(MatrixBase<float> *self, TpMatrix<float> &M){
		self->CopyFromTp(M);
	}

	void AddSp(MatrixBase<float> *self, const float alpha, const SpMatrix<float> &S){
		self->AddSp(alpha, S);
	}

	void AddSpMat(MatrixBase<float> *self, const float alpha, const SpMatrix<float> &A, const MatrixBase<float> &B, MatrixTransposeType transB, const float beta) {
		self->AddSpMat(alpha, A, B, transB, beta);
	}

	void AddTpMat(MatrixBase<float> *self, const float alpha, const TpMatrix<float> &A, MatrixTransposeType transA, const MatrixBase<float> &B, MatrixTransposeType transB, const float beta){
		self->AddTpMat(alpha, A, transA, B, transB, beta);
	}

	void AddMatSp(MatrixBase<float> *self, const float alpha, const MatrixBase<float> &A, MatrixTransposeType transA, const SpMatrix<float> &B, const float beta){
		self->AddMatSp(alpha, A, transA, B, beta);
	}

	void AddMatTp(MatrixBase<float> *self, const float alpha, const MatrixBase<float> &A, MatrixTransposeType transA, const TpMatrix<float> &B, MatrixTransposeType transB, const float beta){
		self->AddMatTp(alpha, A, transA, B, transB, beta);
	}

	void AddTpTp(MatrixBase<float> *self, const float alpha, const TpMatrix<float> &A, MatrixTransposeType transA, TpMatrix<float> &B, MatrixTransposeType transB, const float beta){
		self->AddTpTp(alpha, A, transA, B, transB, beta);
	}

	void AddSpSp(MatrixBase<float> *self, const float alpha, const SpMatrix<float> &A, const SpMatrix<float> &B, const float beta){
		self->AddSpSp(alpha, A, B, beta);
	}

	// void GroupPnormDeriv(MatrixBase<float> *self, MatrixBase<float> &input, float power, MatrixBase<float> &output){
	// 	self->GroupPnormDeriv(input, output, power);
	// }

}


#endif //PYKALDI_MATRIX_KALDI_MATRIX_EXT_H_