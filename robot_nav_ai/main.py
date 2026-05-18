"""Entry point: train or evaluate the robot navigation agent."""

import argparse
import os

from agent.train import train
from agent.evaluate import evaluate

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def parse_args():
    parser = argparse.ArgumentParser(description="Robot Navigation AI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--timesteps", type=int, default=500_000)
    train_parser.add_argument("--n-envs", type=int, default=4)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, default=os.path.join(MODELS_DIR, "ppo_robot_nav_final"))
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--record", action="store_true", help="Record evaluation videos")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)
    elif args.command == "evaluate":
        evaluate(model_path=args.model, n_episodes=args.episodes, record=args.record)


if __name__ == "__main__":
    main()
