"""
Invoice extraction with validation and retry.

The `Invoice` schema (utils/schemas.py) has field constraints (positive quantities, ISO currency code,
VAT rate in [0, 1]) and cross-field validators (line totals, subtotal, VAT and total must add up, due date
after issue date). `extract_with_retry` (utils/retry.py) feeds validation errors back to the model.

Two invoices are processed:
  1. a consistent one - should validate, usually on the first attempt,
  2. one with a typo in the stated total - validation fails; retrying may "fix" the numbers by
     recomputing them, or keep failing. Either way the error messages tell a human reviewer what is wrong,
     which is exactly what you want in a document-processing pipeline.

How to run (from the repo root, with Ollama running):
    python -m p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.main_2_invoice_extraction_with_retry
"""

from pydantic import ValidationError

from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.data import (
    INVOICE_INCONSISTENT,
    INVOICE_OK,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.retry import (
    extract_with_retry,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.s01_structured_outputs.utils.schemas import (
    Invoice,
)
from p07_llms.c02_prompt_engineering_and_structured_outputs.utils.llm import (
    get_chat_model,
)

SYSTEM_PROMPT = (
    "Extract the invoice into the given JSON schema. Copy numbers exactly as written in the document "
    "(without thousands separators). Dates must be ISO formatted (YYYY-MM-DD). "
    "The VAT rate is a fraction (25% -> 0.25)."
)


def demo_validation_without_llm() -> None:
    """
    Show what pydantic validation catches, without any LLM call.
    """
    bad = {
        "invoice_number": "X-1",
        "issue_date": "2025-03-03",
        "seller": "A",
        "buyer": "B",
        "currency": "kroner",
        "line_items": [
            {"description": "work", "quantity": 2, "unit_price": 100, "line_total": 250}
        ],
        "subtotal": 200,
        "vat_rate": 25,
        "vat_amount": 50,
        "total": 250,
    }
    try:
        Invoice.model_validate(bad)
    except ValidationError as e:
        print(f"Pydantic found {e.error_count()} problems in a hand-made bad invoice:")
        for err in e.errors():
            print(
                f"  - {'.'.join(str(p) for p in err['loc']) or '(model)'}: {err['msg']}"
            )


if __name__ == "__main__":
    # 1. Validation is just pydantic - try it on a hand-made bad record
    demo_validation_without_llm()

    # 2. Extract both invoices with retry on validation failure
    llm = get_chat_model(temperature=0)
    for name, text in [
        ("consistent invoice", INVOICE_OK),
        ("invoice with a typo in the total", INVOICE_INCONSISTENT),
    ]:
        print("\n" + "=" * 100)
        print(f"{name}:\n{text}")
        result = extract_with_retry(llm, Invoice, SYSTEM_PROMPT, text, max_attempts=3)
        for i, err in enumerate(result.errors, start=1):
            print(f"Attempt {i} failed: {err[:300]}")
        if result.ok:
            invoice: Invoice = result.parsed
            print(f"Valid after {result.attempts} attempt(s):")
            print(invoice.model_dump_json(indent=2))
        else:
            # 3. In a real pipeline: route to a human review queue together with the error messages
            print(
                f"Still invalid after {result.attempts} attempts -> flag for manual review."
            )
