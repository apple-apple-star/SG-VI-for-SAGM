from __future__ import print_function
from __future__ import absolute_import
from __future__ import division



import os
from argparse import ArgumentParser
import tensorflow as tf
from util import Map
from Run_SGRLD import run_SGRLD

import warnings
from scipy.sparse import SparseEfficiencyWarning


def main(_):
    # Input Arguments
    parser = ArgumentParser()
    parser.add_argument('-seed', dest="seed", type=int, help='Seed of random generator')
    parser.add_argument('-n', '--network', dest="network", help='The name of the network')
    parser.add_argument("-o", "--outfile", dest="outfile", help="The name of the output file")
    parser.add_argument('-lrw', dest="lrw", type=float, help="Initial learning rate for w")
    parser.add_argument('-lrpi', dest="lrpi", type=float, help="Initial learning rate for pi")
    parser.add_argument('-k', dest="K", type=int, help="Number of communities")
    parser.add_argument('-l', dest="L", type=int, help="Mini-batch size")
    parser.add_argument('-m', dest="m", type=int, help="Mini-batch per node")
    parser.add_argument('-ns', dest="num_of_samples_in_records", type=int, help="Number of samples in a records")
    parser.add_argument('-b', '--burn_in', dest="num_burn_in", type=int, help="Number of burn in")
    parser.add_argument('-s', '--num_samples', dest="num_samples", type=int,
                        help="Number of samples collected after burn in")
    parser.add_argument('-tr', '--test_ratio', dest="Test_ratio", type=float, help="Number of Test node pairs to compute AUC")
    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose", default=True,
                        help="don't print status messages to stdout")


    args = parser.parse_args()
    print(args)

    # Parameters of the model
    param_set = Map(beta_0=1., beta_1=1., eta_0=5, eta_1=1, step_size_a_w=args.lrw, step_size_a_pi=args.lrpi,
                    step_size_b=1000, step_size_c=0.55)

    # Create the necessary directory
    if os.path.isdir('../Results') is False:
        os.mkdir('../Results')
    if os.path.isdir('../records') is False:
        os.mkdir('../records')

    warnings.simplefilter('ignore', SparseEfficiencyWarning)

    run_SGRLD(args, param_set)


if __name__ == "__main__":
    tf.app.run()
