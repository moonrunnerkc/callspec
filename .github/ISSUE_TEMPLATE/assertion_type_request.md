---
name: Assertion Type Request
about: Propose a new assertion type for Verdict
title: '[ASSERTION] '
labels: assertion-type
assignees: ''
---

**Assertion category**

Which layer does this belong to? (structural / semantic / behavioral / regression)

**Description**

What does the assertion verify? Be specific about what passes and what fails.

**API design**

How should the assertion look in code?

```python
result = v.assert_that("prompt").your_assertion(params).run()
```

**YAML format**

How should this look in a YAML suite?

```yaml
- type: your_assertion
  params:
    key: value
```

**Scoring method**

How is the assertion evaluated? Binary pass/fail, threshold-based score, or statistical?

**Use case**

What real-world problem does this solve? Include a concrete example.
