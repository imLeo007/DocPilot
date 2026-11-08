from app.core.database import AsyncSessionLocal

from evaluation.evaluator import evaluate_groundforge

from rich.pretty import pprint


def main() -> None:
    results = evaluate_groundforge()

    pprint(results)


if __name__ == "__main__":
    main()