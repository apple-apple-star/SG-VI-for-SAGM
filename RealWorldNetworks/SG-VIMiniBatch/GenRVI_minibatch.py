
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division



import os
from argparse import ArgumentParser
import tensorflow as tf
from util import Map
from Run_Gen_RVI import run_GRVI


def main(_):
    # Input Arguments
    parser = ArgumentParser()
    parser.add_argument('-rc', dest="restart_count", type=int, help="Number of restart count")
    parser.add_argument('-lr_phi', dest="lr_phi", type=float,
                        help="Initial value of learning rate for phi")
    parser.add_argument('-lr_lambda', dest="lr_lambda", type=float,
                        help="Initial value of learning rate for lambda")
    parser.add_argument('-lr_tau', dest="lr_tau", type=float,
                        help="Initial value of learning rate for tau")
    parser.add_argument('-riemann', dest="riemann", type=int, help="1 if use Riemannian Gradient for lambda and phi")
    parser.add_argument('-elbo', dest="elbo", type=int, help="1 if compute estimated ELBO")
    parser.add_argument('-n', '--network', dest="network", help='The name of the network')
    parser.add_argument("-o", "--outfile", dest="outfile", help="The name of the output file")
    parser.add_argument('-k', dest="K", type=int, help="Number of communities")
    parser.add_argument('-mc', dest="mc", type=int, help="Number of monte carlo sample")
    parser.add_argument('-l', dest="L", type=int, help="Mini-batch size")
    parser.add_argument('-m', dest="m", type=int, help="Mini-batch per node")
    parser.add_argument('-ns', dest="num_of_samples_in_records", type=int, help="Number of samples in a records")
    parser.add_argument('-s', '--num_max_iterations', dest="num_max_iterations", type=int,
                        help="Number of maximum iterations")
    parser.add_argument('-tr', '--test_ratio', dest="Test_ratio", type=float,
                        help="Number of Test node pairs to compute AUC")
    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose", default=True,
                        help="don't print status messages to stdout")


    args = parser.parse_args()
    print(args)


    # Parameters of the model
    param_set = Map(beta_0=1., beta_1=1., eta_0=5, eta_1=1,
                    step_size_a_lambda=args.lr_lambda, step_size_a_phi=args.lr_phi, step_size_a_tau=args.lr_tau,
                    step_size_b=1000, step_size_c=0.55,  # Robbin Monroe parameters
                    compute_intermediate_lb=True)  # should lb be computed after each parameter update( for logging purposes)


    # Create the necessary directory
    if os.path.isdir('../Results') is False:
        os.mkdir('../Results')
    if os.path.isdir('../records') is False:
        os.mkdir('../records')

    run_GRVI(args, param_set)

if __name__ == "__main__":
    tf.app.run()
