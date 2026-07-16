import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patch-size",
        type=int,
        choices=[4, 8, 16],
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    for setting in [1, 2, 3, 4]:
        command = [
            sys.executable,
            "train.py",

            "--setting",
            str(setting),

            "--patch-size",
            str(args.patch_size),

            "--epochs",
            str(args.epochs),

            "--batch-size",
            str(args.batch_size),

            "--lr",
            str(args.lr),
        ]

        print()
        print("=" * 70)
        print(
            f"开始运行setting {setting}"
        )
        print(
            f"patch_size={args.patch_size}"
        )
        print(
            f"epochs={args.epochs}"
        )
        print("执行命令：")
        print(" ".join(command))
        print("=" * 70)

        subprocess.run(
            command,
            check=True,
        )

    print()
    print("四种setting已经全部运行完成")


if __name__ == "__main__":
    main()