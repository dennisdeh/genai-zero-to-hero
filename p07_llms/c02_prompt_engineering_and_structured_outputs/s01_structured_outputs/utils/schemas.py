"""
Pydantic schemas used for structured extraction.

The field descriptions are part of the JSON schema that is sent to the model, so they double as
instructions: write them like you would write a prompt.
"""

import datetime as dt
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Financial news extraction (enums, nested models and lists)
# ---------------------------------------------------------------------------
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    MERGER_ACQUISITION = "merger_acquisition"
    MANAGEMENT_CHANGE = "management_change"
    LEGAL_REGULATORY = "legal_regulatory"
    OTHER = "other"


class Company(BaseModel):
    name: str = Field(description="Company name as written in the text")
    ticker: Optional[str] = Field(
        default=None, description="Stock ticker if stated in the text, else null"
    )
    role: Literal["subject", "counterparty", "mentioned"] = Field(
        description="'subject' if the news is mainly about this company"
    )


class FinancialFigure(BaseModel):
    metric: str = Field(
        description="What is measured, e.g. 'revenue', 'operating profit', 'price per share'"
    )
    value: float = Field(description="The number only, e.g. 12.1 for 'DKK 12.1bn'")
    unit: str = Field(description="Unit and scale, e.g. 'DKK bn', 'USD m', '%'")
    period: Optional[str] = Field(
        default=None, description="Period the figure refers to, e.g. 'Q3 2025'"
    )


class NewsExtraction(BaseModel):
    """Structured information extracted from a financial news article."""

    companies: list[Company] = Field(description="All companies mentioned")
    event_type: EventType
    sentiment: Sentiment = Field(
        description="Likely impact on the subject company's share price"
    )
    figures: list[FinancialFigure] = Field(
        default_factory=list, description="All numeric financial figures"
    )
    summary: str = Field(description="One-sentence neutral summary")

    @field_validator("summary")
    @classmethod
    def summary_is_short(cls, value: str) -> str:
        if len(value.split()) > 60:
            raise ValueError("summary must be at most 60 words")
        return value


# ---------------------------------------------------------------------------
# Invoice extraction (dates, cross-field validation)
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0, description="Price per unit excluding VAT")
    line_total: float = Field(ge=0, description="quantity * unit_price, excluding VAT")

    @model_validator(mode="after")
    def check_line_total(self) -> "LineItem":
        if abs(self.quantity * self.unit_price - self.line_total) > 0.01:
            raise ValueError(
                f"line_total {self.line_total} != quantity*unit_price "
                f"({self.quantity}*{self.unit_price}) for '{self.description}'"
            )
        return self


class Invoice(BaseModel):
    """An invoice with consistency checks: the numbers must add up."""

    invoice_number: str
    issue_date: dt.date = Field(description="ISO date YYYY-MM-DD")
    due_date: Optional[dt.date] = Field(default=None, description="ISO date YYYY-MM-DD")
    seller: str
    buyer: str
    currency: str = Field(
        pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. EUR, DKK"
    )
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(description="Sum of line totals excluding VAT")
    vat_rate: float = Field(
        ge=0, le=1, description="VAT rate as a fraction, e.g. 0.25 for 25%"
    )
    vat_amount: float
    total: float = Field(description="Amount to pay including VAT")

    @model_validator(mode="after")
    def check_totals(self) -> "Invoice":
        errors = []
        lines_sum = sum(item.line_total for item in self.line_items)
        if abs(lines_sum - self.subtotal) > 0.01:
            errors.append(
                f"subtotal {self.subtotal} != sum of line totals {lines_sum:.2f}"
            )
        if abs(self.subtotal * self.vat_rate - self.vat_amount) > 0.01:
            errors.append(
                f"vat_amount {self.vat_amount} != subtotal*vat_rate {self.subtotal * self.vat_rate:.2f}"
            )
        if abs(self.subtotal + self.vat_amount - self.total) > 0.01:
            errors.append(
                f"total {self.total} != subtotal+vat_amount {self.subtotal + self.vat_amount:.2f}"
            )
        if self.due_date and self.due_date < self.issue_date:
            errors.append("due_date is before issue_date")
        if errors:
            raise ValueError("; ".join(errors))
        return self
