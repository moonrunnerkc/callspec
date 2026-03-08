# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in LLMAssert, please report it
responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

Email **bradkinnard@proton.me** with:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fix (optional)

You should receive an acknowledgment within 48 hours. A fix or mitigation plan
will be communicated within 7 days of the initial report.

## Scope

Security issues that are in scope for this project include:

- **API key exposure**: Any path where LLMAssert could leak provider API keys
  through logs, error messages, reports, or snapshots
- **Dependency vulnerabilities**: Known CVEs in direct dependencies
  (jsonschema, pyyaml, click, rich, sentence-transformers)
- **Code injection**: Any input (YAML suites, prompts, snapshot files) that
  could lead to arbitrary code execution
- **CI pipeline safety**: Issues in the GitHub Action that could expose secrets
  or allow unauthorized access

## Design Commitments

LLMAssert makes the following security commitments by design:

- **No telemetry, no analytics, no background network traffic.** The library
  makes exactly the LLM API calls you ask for and nothing else.
- **No account required.** No credentials are sent to any LLMAssert-owned
  service.
- **Provider API keys are never logged.** Error messages and reports include
  provider names and model identifiers but never API keys or tokens.
- **Snapshot files contain model outputs only.** No credentials, request
  headers, or authentication tokens are persisted in baseline snapshots.
