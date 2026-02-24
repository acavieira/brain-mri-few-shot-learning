import argparse
from fsl_protonet.config import Config
from fsl_protonet.engine import infer_one_episode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    cfg = Config()
    infer_one_episode(cfg, args.ckpt)