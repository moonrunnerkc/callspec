---
name: Bug Report
about: Report a problem with LLMAssert
title: '[BUG] '
labels: bug
assignees: ''
---

**LLMAssert version:** (output of `llm-assert --version`)

**Python version:** (output of `python --version`)

**Provider:** (e.g., openai, anthropic, mock)

**Description**

A clear description of what went wrong.

**Steps to reproduce**

1. Install llm-assert with `pip install llm-assert[...]`
2. Create test file with:
```python
# minimal reproducer
```
3. Run `pytest` or `llm-assert run ...`

**Expected behavior**

What you expected to happen.

**Actual behavior**

What actually happened. Include the full error output.

**Environment**

- OS: (e.g., Ubuntu 22.04, macOS 14)
- Installation method: pip / source
- Relevant extras installed: (e.g., semantic, openai)
