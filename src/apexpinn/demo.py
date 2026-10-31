"""Entry point for `uv run apexpinn-demo`. The synthetic S1 demo is added in task T5."""


def main() -> None:
    import jax

    print(f"apexpinn demo placeholder. JAX {jax.__version__}, devices: {jax.devices()}")


if __name__ == "__main__":
    main()
