---
name: Bug Report
about: Report a problem with Verdict
title: '[BUG] '
labels: bug
assignees: ''
---

**Verdict version:** (output of `verdict --version`)

**Python version:** (output of `python --version`)

**Provider:** (e.g., openai, anthropic, mock)

**Description**

A clear description of what went wrong.

**Steps to reproduce**

1. Install verdict with `pip install verdict[...]`
2. Create test file with:
```python
# minimal reproducer
```
3. Run `pytest` or `verdict run ...`

**Expected behavior**

What you expected to happen.

**Actual behavior**

What actually happened. Include the full error output.

**Environment**

- OS: (e.g., Ubuntu 22.04, macOS 14)
- Installation method: pip / source
- Relevant extras installed: (e.g., semantic, openai)
