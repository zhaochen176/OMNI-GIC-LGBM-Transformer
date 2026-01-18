import argparse
from pathlib import Path

from src.config import load_config
from src.utils import ensure_dir, get_logger, set_seed
from src.pipelines.run_experiment import run_experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["project"].get("seed", 42)))

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(outputs_dir)

    logger = get_logger(outputs_dir / "logs", cfg["project"]["run_name"])
    logger.info("Config loaded: %s", args.config)

    run_experiment(cfg, logger)

    logger.info("Finished.")


if __name__ == "__main__":
    main()
