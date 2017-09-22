#!/usr/bin/env python

import random
import unittest

from kaldi.base.io import stringstream as sstream
from kaldi.base.io import istringstream, ostringstream
from kaldi.cudamatrix.matrix import ApproxEqualCuMatrix, CuMatrix
from kaldi.matrix import Matrix, Vector
from kaldi.matrix.vector import ApproxEqualVector
from kaldi.nnet3 import *

class TestNnetCompute(unittest.TestCase):

    def test_nnet_compute(self):
        gen_config = NnetGenerationOptions()
        test_collapse_model = random.choice([True, False])

        configs = GenerateConfigSequence(gen_config)
        nnet = Nnet()
        for j, config in enumerate(configs):
            print("Input config[{}]:".format(j))
            print(config)
            istrm = istringstream.new_from_str(config)
            nnet.ReadConfig(istrm)

        request = ComputationRequest()
        inputs = ComputeExampleComputationRequestSimple(nnet, request)
        if test_collapse_model:
            SetBatchnormTestMode(True, nnet)
            SetDropoutTestMode(True, nnet)

        compiler = Compiler(request, nnet)
        opts = CompilerOptions()
        computation = compiler.CreateComputation(opts)

        nnet_collapsed = Nnet.new_from_other(nnet)
        if test_collapse_model:
            collapse_config = CollapseModelConfig()
            CollapseModel(collapse_config, nnet_collapsed)
            compiler_collapsed = Compiler(request, nnet_collapsed)
            computation_collapsed = compiler_collapsed.CreateComputation(opts)
            computation_collapsed.ComputeCudaIndexes()

        ostrm = ostringstream()
        computation.Print(ostrm, nnet)
        print("Generated computation:")
        print(ostrm.to_str())

        check_config = CheckComputationOptions()
        check_config.check_rewrite = True
        checker = ComputationChecker(check_config, nnet, computation)
        checker.Check()

        if random.choice([True, False]):
            opt_config = NnetOptimizeOptions()
            Optimize(opt_config, nnet, MaxOutputTimeInRequest(request),
                     computation)
            ostrm = ostringstream()
            computation.Print(ostrm, nnet)
            print("Optimized computation:")
            print(ostrm.to_str())

        compute_opts = NnetComputeOptions()
        compute_opts.debug = random.choice([True, False])
        computation.ComputeCudaIndexes()
        computer = NnetComputer(compute_opts, computation, nnet, nnet)

        for i, ispec in enumerate(request.inputs):
            temp = CuMatrix.new_from_matrix(inputs[i])
            print("Input sum:", temp.Sum())
            computer.AcceptInput(ispec.name, temp)
        computer.Run()

        output = computer.GetOutputDestructive("output")
        print("Output sum:", output.Sum())

        if test_collapse_model:
            computer_collapsed = NnetComputer(compute_opts,
                                              computation_collapsed,
                                              nnet_collapsed, nnet_collapsed)
            for i, ispec in enumerate(request.inputs):
                temp = CuMatrix.new_from_matrix(inputs[i])
                print("Input sum:", temp.Sum())
                computer_collapsed.AcceptInput(ispec.name, temp)
            computer_collapsed.Run()
            output_collapsed = computer_collapsed.GetOutputDestructive("output")
            print("Output sum [collapsed]:", output_collapsed.Sum())
            self.assertTrue(ApproxEqualCuMatrix(output, output_collapsed),
                            "Regular and collapsed computation outputs differ.")

        output_deriv = CuMatrix.new_from_size(output.NumRows(),
                                              output.NumCols())
        output_deriv.SetRandn()
        if request.outputs[0].has_deriv:
            computer.AcceptInput("output", output_deriv)
            computer.Run()
            for i, ispec in enumerate(request.inputs):
                if ispec.has_deriv:
                    in_deriv = computer.GetOutputDestructive(ispec.name)
                    print("Input-deriv sum for input {} is:".format(ispec.name),
                          in_deriv.Sum())

    def test_nnet_decodable(self):
        gen_config = NnetGenerationOptions()
        configs = GenerateConfigSequence(gen_config)
        nnet = Nnet()
        for j, config in enumerate(configs):
            print("Input config[{}]:".format(j))
            print(config)
            istrm = istringstream.new_from_str(config)
            nnet.ReadConfig(istrm)

        num_frames = 5 + random.randint(1, 100)
        input_dim = nnet.InputDim("input")
        output_dim = nnet.OutputDim("output")
        ivector_dim = max(0, nnet.InputDim("ivector"))
        input = Matrix(num_frames, input_dim)

        SetBatchnormTestMode(True, nnet)
        SetDropoutTestMode(True, nnet)

        input.set_randn()
        ivector = Vector(ivector_dim)
        ivector.set_randn()

        priors = Vector(output_dim if random.choice([True, False]) else 0)
        if len(priors) != 0:
            priors.set_randn()
            priors.apply_exp()

        output1 = Matrix(num_frames, output_dim)
        output2 = Matrix(num_frames, output_dim)

        opts = NnetSimpleComputationOptions()
        opts.frames_per_chunk = random.randint(5, 25)
        compiler = CachingOptimizingCompiler(nnet)
        decodable = DecodableNnetSimple(opts, nnet, priors, input, compiler,
                                        ivector if ivector_dim else None)
        for t in range(num_frames):
            decodable.GetOutputForFrame(t, output1[t])

        opts = NnetSimpleLoopedComputationOptions()
        info = DecodableNnetSimpleLoopedInfo.new_from_priors(opts, priors, nnet)
        decodable = DecodableNnetSimpleLooped(info, input,
                                              ivector if ivector_dim else None)
        for t in range(num_frames):
            decodable.GetOutputForFrame(t, output2[t])

        if (not NnetIsRecurrent(nnet)
            and nnet.Info().find("statistics-extraction") == -1
            and nnet.Info().find("TimeHeightConvolutionComponent") == -1):
            for t in range(num_frames):
                self.assertTrue(ApproxEqualVector(output1[t], output2[t]))


if __name__ == '__main__':
    for i in range(2):
        if cuda_available():
            from kaldi.cudamatrix import CuDevice
            CuDevice.Instantiate().SetDebugStrideMode(True)
            if i == 0:
                CuDevice.Instantiate().SelectGpuId("no")
            else:
                CuDevice.Instantiate().SelectGpuId("yes")
        unittest.main(exit=False)
