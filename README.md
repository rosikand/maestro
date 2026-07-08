# maestro

A minimal post-training library.

## Install

```bash
pip install git+https://github.com/rsikand29/maestro.git
```

Pin a tag or commit for reproducibility:

```bash
pip install git+https://github.com/rsikand29/maestro.git@v0.0.1
```

Confirm it worked:

```bash
python -c "import maestro; maestro.hello()"
# hello, this is maestro
```

## Develop

```bash
pip install -e ".[dev]"
pytest
```
