# Case Study: The Refund Agent That Stopped Checking for Fraud

A customer support agent handles refund requests. The correct behavior is a strict tool-call sequence: verify the customer's identity, check the order history, run a fraud-risk assessment, then issue the refund. Four tools, strict ordering. The fraud check is non-negotiable because skipping it means issuing refunds on stolen accounts.

The agent works correctly on GPT-4o. Every request follows the contract. You ship it.

Three weeks later, your team swaps to a newer model because it is cheaper and faster. All existing tests pass: the agent still responds politely, formats the refund confirmation correctly, handles edge cases. Your text-output tests see no regression.

But the new model skips the fraud-risk tool on refunds under $50. It jumps straight from checking the order to issuing the refund. The agent is now issuing unchecked refunds on small amounts. Your test suite does not catch it because no test asserts on the structure of tool calls.

callspec catches it on the first CI run after the model swap.

## Run it

```bash
pip install callspec
git clone https://github.com/moonrunnerkc/callspec.git
cd callspec
pytest case_study/ -v
```

No API keys. No external services. Under 30 seconds.

## The setup

The refund agent must call four tools in this order on every request:

| Step | Tool | Why |
|------|------|-----|
| 1 | `verify_customer` | Confirm the requester owns the account |
| 2 | `check_order` | Pull the order details and validate the refund amount |
| 3 | `fraud_check` | Assess fraud risk before any money moves |
| 4 | `issue_refund` | Process the refund |

The contract is the same whether the refund is $5 or $5,000. There is no amount threshold that makes fraud checking optional.

## The regression

Two recorded trajectories show the before and after.

**GPT-4o (correct behavior, $32 refund):**

```
[0] verify_customer(customer_id="CUST-1190", verification_method="email")
[1] check_order(order_id="ORD-8842", include_items=true)
[2] fraud_check(order_id="ORD-8842", customer_id="CUST-1190", refund_amount=32.00)
[3] issue_refund(order_id="ORD-8842", amount=32.00, reason="customer_request")
```

**New model (regression, same $32 refund):**

```
[0] verify_customer(customer_id="CUST-1190", verification_method="email")
[1] check_order(order_id="ORD-8842", include_items=true)
[2] issue_refund(order_id="ORD-8842", amount=32.00, reason="customer_request")
```

`fraud_check` is gone. The conversational output still looks correct. The refund confirmation is polite and well-formatted. Text-based tests see nothing wrong.

What makes this regression insidious: on high-value refunds ($340), the new model still calls `fraud_check`. The contract only breaks on small amounts. If you test with one example and happen to pick a high-value refund, you miss it entirely.

## The catch

The callspec test that covers this contract:

```python
REQUIRED_SEQUENCE = [
    "verify_customer",
    "check_order",
    "fraud_check",
    "issue_refund",
]

def test_refund_contract(trajectory):
    result = (
        TrajectoryBuilder(trajectory)
        .calls_tools_in_order(REQUIRED_SEQUENCE)
        .calls_tool("fraud_check")
        .call_count("fraud_check", min_count=1, max_count=1)
        .argument_not_empty("fraud_check", "order_id")
        .run()
    )
    assert result.passed
```

When run against the regressed trajectory, callspec produces:

```
[FAIL] calls_tools_in_order: Expected tools in order ['verify_customer',
       'check_order', 'fraud_check', 'issue_refund']. Matched
       ['verify_customer', 'check_order'] but 'fraud_check' not found
       after position 2. Actual trajectory: ['verify_customer',
       'check_order', 'issue_refund'].

[FAIL] calls_tool: Tool 'fraud_check' not found in trajectory. Tools
       called: ['verify_customer', 'check_order', 'issue_refund'].

[FAIL] call_count: Tool 'fraud_check' called 0 time(s), expected
       range [1, 1].

[FAIL] argument_not_empty: Cannot check emptiness for key 'order_id':
       tool 'fraud_check' not found in trajectory.
```

The snapshot drift report shows the structural change:

```
Sequence match:  False
Tools removed:   ['fraud_check']

Baseline: ['verify_customer', 'check_order', 'fraud_check', 'issue_refund']
Current:  ['verify_customer', 'check_order', 'issue_refund']

Model: gpt-4o-2024-11-20 -> gpt-4.1-2025-04-14
```

That is what appears in CI on the first run after the model swap. Before the change ships.

## What is in this directory

```
case_study/
  README.md                               # this file
  test_refund_agent.py                    # 11 pytest tests covering all angles
  trajectories/
    gpt4o_correct.json                    # baseline: correct 4-tool sequence
    gpt4o_high_value.json                 # baseline: $340 refund, same contract
    new_model_regression.json             # regression: fraud_check skipped ($32)
    new_model_high_value.json             # no regression on $340 (the trap)
  snapshots/
    baselines.json                        # recorded baseline for drift detection
```

The test file has four sections:

1. **TestBaselineBehavior** -- confirms GPT-4o follows the contract on both low and high value refunds.
2. **TestNewModelRegression** -- demonstrates the regression: low-value refund fails, high-value still passes.
3. **TestSnapshotDrift** -- shows snapshot baselines catching the drift automatically.
4. **TestFullContract** -- the complete production contract with ordering, counts, forbidden tools, and argument validation.

## A note on the trajectories

These trajectories are constructed from documented behavioral patterns observed across model versions. They are not captured from a live API call in this repository. In production, you would capture trajectories from your actual agent runs using `ToolCallTrajectory.from_provider_response(response)` and commit them as snapshot baselines.

The point of this case study is not that this specific regression happened on this specific model. The point is that when a regression like this happens, and it will, callspec catches it before it ships.

## The pattern generalizes

Nearly every production agent has a tool call that must not be skipped:

- Medical triage agents that must call a contraindication check
- Financial agents that must verify account ownership before transfers
- Content moderation agents that must run a policy check before publishing
- E-commerce agents that must validate inventory before confirming orders

If you have an agent in production with no coverage on tool-call ordering, you have an untested contract.
