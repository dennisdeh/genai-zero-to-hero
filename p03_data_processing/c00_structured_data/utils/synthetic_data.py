"""
Synthetic "customers & transactions" data used throughout the structured-data chapter.

Real-world tabular data is rarely clean. To practise cleaning, feature engineering and data quality
assurance without depending on downloads or accounts, this module generates a realistic dataset of
an (imaginary) online shop:

- customers:    one row per customer with demographics, plan, income, support tickets and a churn label
- transactions: one row per purchase, linked to customers via ``customer_id``

The data is first generated *clean* (with sensible dtypes and a known ground truth) by
``generate_clean_data`` and can then be *corrupted* by ``make_messy`` which introduces the typical
problems found in practice:

- messy column names, mixed date formats, numbers stored as text with currency symbols
- inconsistent spelling/casing of categories, missing values, invalid values and outliers
- exact and near duplicates, orphaned foreign keys (transactions of unknown customers)

``generate_raw_data`` combines both and round-trips the result through CSV, so the raw data looks
exactly like a freshly read CSV export (all columns are text).

The ``drift`` argument shifts the distributions (older, richer customers, more electronics), which is
used to simulate a new data batch in the data quality assurance section.

Usage (from the repository root):
    from p03_data_processing.c00_structured_data.utils.synthetic_data import generate_raw_data
    customers_raw, transactions_raw = generate_raw_data(n_customers=2000, seed=42)
"""

import io

import numpy as np
import pandas as pd

# canonical values of the categorical columns
COUNTRIES = ["Denmark", "Germany", "Sweden", "Norway", "Netherlands"]
SEGMENTS = ["Consumer", "Small Business", "Corporate"]
PLANS = ["Basic", "Standard", "Premium"]  # ordinal: Basic < Standard < Premium
CATEGORIES = ["Electronics", "Groceries", "Clothing", "Home", "Sports"]
PAYMENT_METHODS = ["card", "mobile_pay", "invoice", "bank_transfer"]

# typical spelling variants seen in raw data (canonical value -> variants)
COUNTRY_VARIANTS = {
    "Denmark": ["denmark", "DK", " Denmark ", "DENMARK", "Danmark"],
    "Germany": ["germany", "DE", "Deutschland", "GERMANY "],
    "Sweden": ["sweden", "SE", "Sverige"],
    "Norway": ["norway", "NO", "Norge"],
    "Netherlands": ["netherlands", "NL", "The Netherlands", "Holland"],
}
SEGMENT_VARIANTS = {
    "Consumer": ["consumer", "CONSUMER", "Consumer "],
    "Small Business": ["small business", "Small_Business", "SMB", "small-business"],
    "Corporate": ["corporate", "Corp.", "CORPORATE"],
}
CATEGORY_VARIANTS = {
    "Electronics": ["electronics", "Electronic", "ELECTRONICS "],
    "Groceries": ["groceries", "Grocery", " Groceries"],
    "Clothing": ["clothing", "Clothes", "CLOTHING"],
    "Home": ["home", "Home & Garden", "HOME"],
    "Sports": ["sports", "Sport", "SPORTS"],
}

# messy column names as they could come from a spreadsheet export
MESSY_CUSTOMER_COLUMNS = {
    "customer_id": "Customer ID",
    "signup_date": "Signup Date ",
    "age": "AGE",
    "country": "country",
    "segment": "Segment",
    "plan": "Plan",
    "annual_income": "Annual Income (EUR)",
    "support_tickets": "Support Tickets",
    "email": "E-mail",
    "churned": "Churned?",
}
MESSY_TRANSACTION_COLUMNS = {
    "transaction_id": "Transaction ID",
    "customer_id": "Customer ID",
    "timestamp": "Timestamp",
    "amount": "Amount",
    "category": "Category",
    "payment_method": "Payment Method",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def _shifted_probs(probs: list[float], drift: float, favoured_index: int) -> np.ndarray:
    """Move probability mass towards one category proportionally to ``drift`` (0 = unchanged)."""
    p = np.asarray(probs, dtype=float)
    p[favoured_index] += drift * 0.5
    return p / p.sum()


def generate_clean_data(
    n_customers: int = 2000,
    seed: int = 42,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    drift: float = 0.0,
    id_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a clean customers and transactions dataset with a known data-generating process.

    A latent "engagement" score drives the number of purchases, the number of support tickets and
    the churn probability, so the churn label can be predicted from engineered features.

    :param n_customers: number of customers to generate
    :param seed: random seed for reproducibility
    :param start_date: first possible signup/transaction date
    :param end_date: last possible signup/transaction date (end of the observation window)
    :param drift: strength of the distribution shift (0 = reference distribution)
    :param id_offset: offset added to the customer/transaction numbers (to keep ids unique across batches)
    :return: (customers, transactions) data frames with proper dtypes
    """
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    window_days = (end - start).days

    # 1. customer attributes
    engagement = rng.normal(0.0, 1.0, n_customers)
    age = np.clip(rng.normal(42 + 8 * drift, 12, n_customers), 18, 90).round().astype(int)
    country = rng.choice(COUNTRIES, n_customers, p=_shifted_probs([0.35, 0.25, 0.15, 0.15, 0.10], drift, 1))
    segment = rng.choice(SEGMENTS, n_customers, p=[0.60, 0.25, 0.15])
    segment_effect = np.select([segment == "Corporate", segment == "Small Business"], [0.35, 0.15], 0.0)
    income = np.exp(rng.normal(np.log(48_000) + segment_effect + 0.25 * drift, 0.45, n_customers))
    income = (income / 100).round() * 100
    # richer customers and businesses choose more expensive plans
    plan_score = 0.8 * (np.log(income) - np.log(48_000)) + segment_effect + rng.normal(0, 0.6, n_customers)
    plan = np.select([plan_score > 0.6, plan_score > -0.2], ["Premium", "Standard"], "Basic")
    signup_offset = rng.integers(0, window_days - 60, n_customers)
    signup_date = start + pd.to_timedelta(signup_offset, unit="D")
    support_tickets = rng.poisson(np.exp(0.2 - 0.5 * engagement))
    domains = rng.choice(["example.com", "example.org", "example.net"], n_customers)
    customer_numbers = np.arange(1, n_customers + 1) + id_offset
    customer_id = [f"C{n:06d}" for n in customer_numbers]
    email = [f"user{n}@{d}" for n, d in zip(customer_numbers, domains)]

    # 2. churn label from a logistic model of the (partly latent) attributes
    logit = (
        -1.6
        - 1.1 * engagement
        + 0.35 * support_tickets
        - 0.4 * (plan == "Premium")
        + 0.3 * (segment == "Consumer")
        - 0.01 * (age - 42)
    )
    churned = rng.random(n_customers) < _sigmoid(logit)

    customers = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": signup_date,
            "age": age,
            "country": pd.Categorical(country, categories=COUNTRIES),
            "segment": pd.Categorical(segment, categories=SEGMENTS),
            "plan": pd.Categorical(plan, categories=PLANS, ordered=True),
            "annual_income": income,
            "support_tickets": support_tickets,
            "email": email,
            "churned": churned,
        }
    )

    # 3. transactions: engaged customers buy more often; churned customers stop buying at some point
    n_tx = rng.poisson(np.exp(1.6 + 0.5 * engagement))
    active_days = (end - signup_date).days.to_numpy()
    churn_after = np.where(churned, rng.uniform(0.1, 0.7, n_customers) * active_days, active_days)
    cat_probs = _shifted_probs([0.20, 0.30, 0.20, 0.15, 0.15], drift, 0)
    mean_amount = {"Electronics": 250.0, "Groceries": 45.0, "Clothing": 70.0, "Home": 120.0, "Sports": 90.0}

    owner = np.repeat(np.arange(n_customers), n_tx)
    n_total = len(owner)
    tx_offset_days = rng.uniform(0, 1, n_total) * churn_after[owner]
    timestamp = (signup_date[owner] + pd.to_timedelta(tx_offset_days, unit="D")).floor("min")
    category = rng.choice(CATEGORIES, n_total, p=cat_probs)
    base = np.array([mean_amount[c] for c in category])
    amount = np.round(base * np.exp(rng.normal(-0.18, 0.6, n_total)), 2)
    payment = rng.choice(PAYMENT_METHODS, n_total, p=[0.55, 0.25, 0.10, 0.10])

    transactions = pd.DataFrame(
        {
            "transaction_id": [f"T{n:08d}" for n in np.arange(1, n_total + 1) + id_offset * 10],
            "customer_id": np.asarray(customer_id)[owner],
            "timestamp": timestamp,
            "amount": amount,
            "category": pd.Categorical(category, categories=CATEGORIES),
            "payment_method": pd.Categorical(payment, categories=PAYMENT_METHODS),
        }
    )
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)
    return customers, transactions


def _format_dates(dates: pd.Series, formats: list[str], probs: list[float], rng: np.random.Generator) -> list:
    """Format each date with a randomly chosen format string (simulates data from several sources)."""
    chosen = rng.choice(len(formats), len(dates), p=probs)
    return [d.strftime(formats[i]) if pd.notna(d) else None for d, i in zip(dates, chosen)]


def _random_variant(value: str, variants: dict[str, list[str]], rng: np.random.Generator) -> str:
    """Return a random spelling variant of a canonical category value."""
    return str(rng.choice(variants[value]))


def _blank_out(values: pd.Series, frac: float, rng: np.random.Generator) -> pd.Series:
    """Set a random fraction of the values to missing."""
    values = values.astype(object).copy()
    mask = rng.random(len(values)) < frac
    values[mask] = None
    return values


def make_messy(
    customers: pd.DataFrame, transactions: pd.DataFrame, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corrupt clean data with typical real-world data quality problems.

    :param customers: clean customers data frame (from ``generate_clean_data``)
    :param transactions: clean transactions data frame
    :param seed: random seed
    :return: (customers_raw, transactions_raw) with messy names, formats, values and duplicates
    """
    rng = np.random.default_rng(seed)
    c = customers.copy().astype(object)
    n = len(c)

    # 1. dates in mixed formats (+ a few unparsable entries)
    c["signup_date"] = _format_dates(
        customers["signup_date"], ["%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%B %d, %Y"], [0.6, 0.2, 0.1, 0.1], rng
    )
    c.loc[rng.random(n) < 0.01, "signup_date"] = "not available"

    # 2. invalid ages (sentinel values and typos)
    invalid_age = rng.random(n) < 0.015
    c.loc[invalid_age, "age"] = rng.choice([-1, 0, 999, 420], invalid_age.sum())

    # 3. inconsistent categories
    for col, variants, frac in [("country", COUNTRY_VARIANTS, 0.30), ("segment", SEGMENT_VARIANTS, 0.25)]:
        mask = rng.random(n) < frac
        c.loc[mask, col] = [_random_variant(v, variants, rng) for v in c.loc[mask, col]]
    mask = rng.random(n) < 0.10
    c.loc[mask, "plan"] = c.loc[mask, "plan"].str.lower()

    # 4. income stored as text with thousands separators/currency; a few data-entry errors (x50)
    income = customers["annual_income"].to_numpy().copy()
    entry_error = rng.random(n) < 0.01
    income[entry_error] *= 50
    style = rng.choice(3, n, p=[0.75, 0.15, 0.10])
    c["annual_income"] = [
        f"{v:.1f}" if s == 0 else (f"{v:,.0f}" if s == 1 else f"EUR {v:.0f}") for v, s in zip(income, style)
    ]

    # 5. emails: some invalid
    bad_email = rng.random(n) < 0.03
    c.loc[bad_email, "email"] = c.loc[bad_email, "email"].str.replace("@", "", regex=False)

    # 6. churn label in mixed boolean encodings
    c["churned"] = [
        str(rng.choice(["yes", "1", "True"])) if v else str(rng.choice(["no", "0", "False"]))
        for v in customers["churned"]
    ]

    # 7. missing values
    for col, frac in [("age", 0.05), ("country", 0.02), ("annual_income", 0.05), ("email", 0.03),
                      ("support_tickets", 0.02), ("signup_date", 0.01)]:
        c[col] = _blank_out(c[col], frac, rng)

    # 8. exact duplicates and near duplicates (same customer, some fields blanked)
    exact = c.sample(frac=0.03, random_state=seed)
    near = c.sample(frac=0.01, random_state=seed + 1).copy()
    near[["age", "annual_income"]] = None
    c = pd.concat([c, exact, near], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    c = c.rename(columns=MESSY_CUSTOMER_COLUMNS)

    # --- transactions ---
    t = transactions.copy().astype(object)
    m = len(t)
    t["timestamp"] = _format_dates(
        transactions["timestamp"], ["%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M:%S"], [0.7, 0.15, 0.15], rng
    )
    amount = transactions["amount"].to_numpy().copy()
    refunds = rng.random(m) < 0.01
    amount[refunds] *= -1  # refunds are legitimate negative amounts
    extreme = rng.random(m) < 0.005
    amount[extreme] *= 100  # extreme outliers (e.g. a wrong unit or a bulk order)
    with_symbol = rng.random(m) < 0.10
    t["amount"] = [f"€{v:.2f}" if s else f"{v:.2f}" for v, s in zip(amount, with_symbol)]
    mask = rng.random(m) < 0.25
    t.loc[mask, "category"] = [_random_variant(v, CATEGORY_VARIANTS, rng) for v in t.loc[mask, "category"]]
    t["payment_method"] = _blank_out(t["payment_method"], 0.05, rng)
    orphans = rng.random(m) < 0.01
    t.loc[orphans, "customer_id"] = [f"C9{k:05d}" for k in rng.integers(0, 99_999, orphans.sum())]
    t = pd.concat([t, t.sample(frac=0.02, random_state=seed)], ignore_index=True)
    t = t.sample(frac=1.0, random_state=seed).reset_index(drop=True).rename(columns=MESSY_TRANSACTION_COLUMNS)
    return c, t


def _csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Write a frame to CSV in memory and read it back as text (like reading a raw export)."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype=str, keep_default_na=True)


def generate_raw_data(
    n_customers: int = 2000, seed: int = 42, drift: float = 0.0, id_offset: int = 0, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate messy raw data as it would be read from CSV exports (all columns are text).

    :param n_customers: number of customers
    :param seed: random seed
    :param drift: distribution shift (see ``generate_clean_data``)
    :param id_offset: offset for the ids
    :param kwargs: further arguments passed to ``generate_clean_data`` (e.g. start_date, end_date)
    :return: (customers_raw, transactions_raw)
    """
    customers, transactions = generate_clean_data(
        n_customers=n_customers, seed=seed, drift=drift, id_offset=id_offset, **kwargs
    )
    customers_raw, transactions_raw = make_messy(customers, transactions, seed=seed + 1)
    return _csv_roundtrip(customers_raw), _csv_roundtrip(transactions_raw)


if __name__ == "__main__":
    cust, tx = generate_clean_data(n_customers=500)
    print(cust.head(), "\n", tx.head(), "\n", f"churn rate: {cust['churned'].mean():.1%}")
    cust_raw, tx_raw = generate_raw_data(n_customers=500)
    print(cust_raw.head(10).to_string())
    print(tx_raw.head(10).to_string())
