"""Loading and cleaning steps for the Online Retail II dataset.

Each step is a small function so the notebook can run them one at a time and
log what every step removes. The order matters and is documented in
notebooks/01_data_cleaning.ipynb.
"""

from pathlib import Path

import pandas as pd

SHEETS = ("Year 2009-2010", "Year 2010-2011")

COLUMN_NAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Stock codes that are not merchandise: shipping, marketplace and bank fees,
# manual adjustments, discounts, samples, charity commission, test items and
# gift vouchers. The DCGS* codes are real products sold on the web shop and stay.
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "C3",          # postage and carriage
    "AMAZONFEE", "BANK CHARGES", "CRUK",  # fees and commissions
    "M", "m", "ADJUST", "ADJUST2", "B",   # manual entries and accounting adjustments
    "D", "S",                             # discounts and samples
    "TEST001", "TEST002",                 # test products
}
NON_PRODUCT_PREFIXES = ("gift_", "GIFT")  # gift vouchers


def load_sheets(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read both yearly sheets as they are, keeping codes and invoices as text."""
    # Some invoice numbers start with a letter and some descriptions are numbers,
    # so these columns are forced to str instead of letting pandas guess
    return pd.read_excel(
        path,
        sheet_name=list(SHEETS),
        dtype={"Invoice": str, "StockCode": str, "Description": str},
    )


def combine_sheets(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the two sheets, dropping the days that appear in both.

    The second sheet starts on 2010-12-01 while the first one ends on
    2010-12-09: those nine days are exported twice, so they are taken only
    from the first sheet.
    """
    first, second = (sheets[name] for name in SHEETS)
    last_day_first = first["InvoiceDate"].max()
    second = second[second["InvoiceDate"] > last_day_first]
    df = pd.concat([first, second], ignore_index=True)
    return df.rename(columns=COLUMN_NAMES)


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows identical in every column (same invoice, item, quantity, price, minute)."""
    return df.drop_duplicates()


def drop_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only lines that can be attributed to a customer, with an integer id."""
    df = df[df["customer_id"].notna()].copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    return df


def drop_non_positive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove free items (price 0) and accounting entries with a negative price."""
    return df[df["price"] > 0]


def drop_non_products(df: pd.DataFrame) -> pd.DataFrame:
    """Remove lines whose stock code is a fee, adjustment, test item or voucher."""
    code = df["stock_code"]
    is_non_product = code.isin(NON_PRODUCT_CODES) | code.str.startswith(NON_PRODUCT_PREFIXES)
    return df[~is_non_product]


def split_purchases_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split lines into purchases and returns (cancelled invoices start with 'C')."""
    is_return = df["invoice"].str.startswith("C")
    purchases = df[~is_return & (df["quantity"] > 0)].copy()
    returns = df[is_return & (df["quantity"] < 0)].copy()
    for part in (purchases, returns):
        part["revenue"] = part["quantity"] * part["price"]
    return purchases, returns


def cancel_matched_returns(
    purchases: pd.DataFrame, returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Remove returns that exactly undo an earlier purchase line, together with that line.

    A return matches a purchase when customer, stock code, price and quantity
    (in absolute value) are the same and the purchase is not later than the
    return. Each purchase line can be cancelled at most once: a return is
    paired with the most recent eligible purchase still unmatched.

    Returns the purchases and returns left after matching, and the number of pairs.
    """
    keys = ["customer_id", "stock_code", "price", "abs_quantity"]
    p = purchases.assign(abs_quantity=purchases["quantity"])
    r = returns.assign(abs_quantity=-returns["quantity"])

    # Only purchase lines sharing a key with some return can ever be matched
    candidates = p.reset_index(names="line_id").merge(r[keys].drop_duplicates(), on=keys)
    candidates = candidates.sort_values("invoice_date")
    open_purchases = {
        key: list(zip(group["invoice_date"], group["line_id"]))
        for key, group in candidates.groupby(keys)
    }

    matched_purchases, matched_returns = [], []
    for line_id, row in r.sort_values("invoice_date").iterrows():
        options = open_purchases.get(tuple(row[keys]), [])
        # Walk backwards to find the latest purchase not after the return
        for i in range(len(options) - 1, -1, -1):
            purchase_date, purchase_id = options[i]
            if purchase_date <= row["invoice_date"]:
                matched_purchases.append(purchase_id)
                matched_returns.append(line_id)
                del options[i]
                break

    return (
        purchases.drop(index=matched_purchases),
        returns.drop(index=matched_returns),
        len(matched_returns),
    )
