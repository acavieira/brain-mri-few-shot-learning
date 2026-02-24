from fsl_protonet.config import Config
from fsl_protonet.engine import train

if __name__ == "__main__":
    cfg = Config()
    train(cfg)