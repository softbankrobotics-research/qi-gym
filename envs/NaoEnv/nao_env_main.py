import argparse
import time
import numpy as np
import baselines_tools


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generate_pretrain",
        type=int,
        default=0,
        help="If true, launch an interface to generate an expert trajectory")

    parser.add_argument(
        "--train",
        type=int,
        default=0,
        help="True: training, False: nothing")

    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="True: pretrainning, False: nothing")

    parser.add_argument(
        "--enjoy",
        type=int,
        default=0,
        help="True: enjoy pretrained model, False: nothing")

    args = parser.parse_args()

    if args.generate_pretrain:
        baselines_tools.collect_pretrained_dataset(baselines_tools.MODEL_NAME)

    if args.train:
        seed = int(time.time())
        np.random.seed(seed)
        # train the model
        baselines_tools.train(num_timesteps=int(1e7), seed=seed,
                              model_path=baselines_tools.PATH_MODEL)

    if args.pretrain:
        baselines_tools.pretrained_model_and_save(baselines_tools.MODEL_NAME)

    if args.enjoy:
        baselines_tools.visualize(
            baselines_tools.PATH_MODEL +
            baselines_tools.AGENT + "_" + baselines_tools.MODEL_NAME)


if __name__ == "__main__":
    main()
