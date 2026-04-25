import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'uvicorn'. Install it with: python -m pip install uvicorn"
        ) from exc

    # Prefer passing the app object directly to avoid import path issues on Windows.
    from ml_system.api import app  # type: ignore

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
