# llm-assert Deprecation Release

This directory contains the minimal package for publishing `llm-assert` v0.1.1 as a deprecation notice on PyPI.

## What This Does

When someone runs `pip install llm-assert`, they get v0.1.1 which:
1. Prints a deprecation warning on import
2. Re-exports the `callspec` package so existing code continues working
3. Lists `callspec` as a dependency so it gets installed automatically

## How to Publish

```bash
cd deprecation
python -m build
twine upload dist/*
```

Or push a `v0.1.1` tag from a branch that uses this pyproject.toml instead of the main one.

## After Publishing

Do NOT yank the old v0.1.0 from PyPI. Pinned installs should continue to work.
