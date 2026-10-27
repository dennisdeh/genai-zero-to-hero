"""
Small, fictional example texts used by the structured output examples (no download needed).
"""

NEWS_ARTICLES = [
    (
        "Nordic Wind A/S (ticker: NWIND) reported third-quarter revenue of DKK 4.2bn, up 18% year on year, "
        "while operating profit rose to DKK 610m from DKK 450m. The company raised its full-year revenue "
        "guidance to DKK 16-17bn. Shares rose 6% in early trading in Copenhagen."
    ),
    (
        "Atlantic Freight Ltd agreed to acquire the logistics start-up ParcelPilot for USD 350m in cash, "
        "a premium of 30% to its last funding round. Atlantic Freight expects the deal to close in Q2 2026 "
        "and to add USD 40m in annual revenue."
    ),
    (
        "The financial regulator fined Helios Bank EUR 25m for weaknesses in its anti-money-laundering "
        "controls between 2019 and 2023. Helios Bank said it accepts the decision and has already "
        "strengthened its compliance function. The shares fell 3%."
    ),
    (
        "Greenleaf Foods announced that chief executive Maria Jensen will step down at the end of the year "
        "after eight years in the role. Chief operating officer Lars Holm will succeed her. Greenleaf "
        "reiterated its outlook of 5% organic growth for 2025."
    ),
]

# A consistent invoice: all numbers add up
INVOICE_OK = """
INVOICE  No. 2025-0142
Date: 3 March 2025          Payment due: 2 April 2025
From: Hansen Data Consulting ApS, Copenhagen
To:   Fjord Analytics A/S, Aarhus

  Description                          Qty   Unit price (DKK)   Amount (DKK)
  Data pipeline review (hours)          12        1,100.00        13,200.00
  Model monitoring set-up (fixed)        1        8,500.00         8,500.00
  Travel                                 2          450.00           900.00

  Subtotal                                                       22,600.00
  VAT 25%                                                         5,650.00
  TOTAL DKK                                                      28,250.00
"""

# An inconsistent invoice: the stated total contains a typo (should be EUR 555.00).
# Validation should catch this; retrying cannot fix an error in the *source* document.
INVOICE_INCONSISTENT = """
Invoice #A-77
Issued 2025-05-10 by Café Bønne, Odense, to Hygge Events ApS.
3 x catering platter at EUR 120.00 = EUR 360.00
24 x coffee at EUR 3.50 = EUR 84.00
Subtotal EUR 444.00, VAT (25%) EUR 111.00, Total EUR 565.00
"""
