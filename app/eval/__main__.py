from __future__ import annotations

import asyncio
import json

from ..db import connect, disconnect
from .runner import run_eval


async def _main() -> None:
    await connect()
    try:
        report = await run_eval()
        print(json.dumps(report["summary"], indent=2))
        print("\nPer-case:")
        for case in report["cases"]:
            scores = {n: m["score"] for n, m in case["metrics"].items()}
            print(f"  [{case['kind']}] {case['question'][:70]}... -> {scores}")
    finally:
        await disconnect()


# Run inside the container:  docker compose exec app python -m app.eval
if __name__ == "__main__":
    asyncio.run(_main())
