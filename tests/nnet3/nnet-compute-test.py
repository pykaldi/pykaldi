#!/usr/bin/env python

import random
import unittest

from kaldi.base.io import istringstream, ostringstream
from kaldi.cudamatrix import cuda_available, approx_equal_cu_matrix, CuMatrix
from kaldi.matrix import Matrix, Vector
from kaldi.matrix.functions import approx_equal
from kaldi.nnet3 import *

class TestNnetCompute(unittest.TestCase):

    def test_nnet_compute(self):
        gen_config = NnetGenerationOptions()
        test_collapse_model = random.choice([True, False])

        configs = generate_config_sequence(gen_config)
        nnet = Nnet()
        for j, config in enumerate(configs):
            # print("Input config[{}]:".format(j))
            # print(config)
            istrm = istringstream.from_str(config)
            nnet.read_config(istrm)

        request = ComputationRequest()
        inputs = compute_example_computation_request_simple(nnet, request)
        if test_collapse_model:
            set_batchnorm_test_mode(True, nnet)
            set_dropout_test_mode(True, nnet)

        compiler = Compiler(request, nnet)
        opts = CompilerOptions()
        computation = compiler.create_computation(opts)

        nnet_collapsed = Nnet.from_other(nnet)
        if test_collapse_model:
            collapse_config = CollapseModelConfig()
            collapse_model(collapse_config, nnet_collapsed)
            compiler_collapsed = Compiler(request, nnet_collapsed)
            computation_collapsed = compiler_collapsed.create_computation(opts)
            computation_collapsed.compute_cuda_indexes()

        ostrm = ostringstream()
        computation.print_computation(ostrm, nnet)
        # print("Generated computation:")
        # print(ostrm.to_str())

        check_config = CheckComputationOptions()
        check_config.check_rewrite = True
        checker = ComputationChecker(check_config, nnet, computation)
        checker.check()

        if random.choice([True, False]):
            opt_config = NnetOptimizeOptions()
            optimize(opt_config, nnet, max_output_time_in_request(request),
                     computation)
            ostrm = ostringstream()
            computation.print_computation(ostrm, nnet)
            # print("Optimized computation:")
            # print(ostrm.to_str())

        compute_opts = NnetComputeOptions()
        compute_opts.debug = random.choice([True, False])
        computation.compute_cuda_indexes()
        computer = NnetComputer(compute_opts, computation, nnet, nnet)

        for i, ispec in enumerate(request.inputs):
            temp = CuMatrix.from_matrix(inputs[i])
            print("Input sum:", temp.sum())
            computer.accept_input(ispec.name, temp)
        computer.run()

        output = computer.get_output_destructive("output")
        print("Output sum:", output.sum())

        if test_collapse_model:
            computer_collapsed = NnetComputer(compute_opts,
                                              computation_collapsed,
                                              nnet_collapsed, nnet_collapsed)
            for i, ispec in enumerate(request.inputs):
                temp = CuMatrix.from_matrix(inputs[i])
                computer_collapsed.accept_input(ispec.name, temp)
            computer_collapsed.run()
            output_collapsed = computer_collapsed.get_output_destructive("output")
            print("Output sum [collapsed]:", output_collapsed.sum())
            self.assertTrue(approx_equal_cu_matrix(output, output_collapsed),
                            "Regular and collapsed computation outputs differ.")

        output_deriv = CuMatrix.from_size(output.num_rows(), output.num_cols())
        output_deriv.set_randn()
        if request.outputs[0].has_deriv:
            computer.accept_input("output", output_deriv)
            computer.run()
            for i, ispec in enumerate(request.inputs):
                if ispec.has_deriv:
                    in_deriv = computer.get_output_destructive(ispec.name)
                    print("Input-deriv sum for input {} is:".format(ispec.name),
                          in_deriv.sum())

    def test_nnet_decodable(self):
        gen_config = NnetGenerationOptions()
        configs = generate_config_sequence(gen_config)
        nnet = Nnet()
        for j, config in enumerate(configs):
            # print("Input config[{}]:".format(j))
            # print(config)
            istrm = istringstream.from_str(config)
            nnet.read_config(istrm)

        num_frames = 5 + random.randint(1, 100)
        input_dim = nnet.input_dim("input")
        output_dim = nnet.output_dim("output")
        ivector_dim = max(0, nnet.input_dim("ivector"))
        input = Matrix(num_frames, input_dim)

        set_batchnorm_test_mode(True, nnet)
        set_dropout_test_mode(True, nnet)

        input.set_randn_()
        ivector = Vector(ivector_dim)
        ivector.set_randn_()

        priors = Vector(output_dim if random.choice([True, False]) else 0)
        if len(priors) != 0:
            priors.set_randn_()
            priors.apply_exp_()

        output1 = Matrix(num_frames, output_dim)
        output2 = Matrix(num_frames, output_dim)

        opts = NnetSimpleComputationOptions()
        opts.frames_per_chunk = random.randint(5, 25)
        compiler = CachingOptimizingCompiler(nnet)
        decodable = DecodableNnetSimple(opts, nnet, priors, input, compiler,
                                        ivector if ivector_dim else None)
        for t in range(num_frames):
            decodable.get_output_for_frame(t, output1[t])

        opts = NnetSimpleLoopedComputationOptions()
        info = DecodableNnetSimpleLoopedInfo.from_priors(opts, priors, nnet)
        decodable = DecodableNnetSimpleLooped(info, input,
                                              ivector if ivector_dim else None)
        for t in range(num_frames):
            decodable.get_output_for_frame(t, output2[t])

        if (not nnet_is_recurrent(nnet)
            and nnet.info().find("statistics-extraction") == -1
            and nnet.info().find("TimeHeightConvolutionComponent") == -1
            and nnet.info().find("RestrictedAttentionComponent") == -1):
            for t in range(num_frames):
                self.assertTrue(approx_equal(output1[t], output2[t]))


if __name__ == '__main__':
    for i in range(2):
        if cuda_available():
            from kaldi.cudamatrix import CuDevice
            CuDevice.instantiate().set_debug_stride_mode(True)
            if i == 0:
                CuDevice.instantiate().select_gpu_id("no")
            else:
                CuDevice.instantiate().select_gpu_id("yes")
        unittest.main(exit=False)
