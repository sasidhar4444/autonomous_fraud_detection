# scripts/generate_dirty_test_data.py
import csv
import random
from datetime import datetime, timedelta
import uuid

OUT = "data/test_dirty_transactions.csv"
N = 2000  # size; change to 50k if you want stress test

merchants = ["flipkart", "amazon", "myntra", "paytm", "zomato", "bigbasket", "local_shop"]
methods = ["upi", "card", "netbanking"]
countries = ["IN", "US", "AE", "GB", "NG"]

def random_timestamp(bad_rate=0.1):
    if random.random() < bad_rate:
        # produce dirty timestamp
        choices = ["not-a-date", "", "2025/13/01 99:99:99"]
        return random.choice(choices)
    # good timestamp within last 90 days
    t = datetime.utcnow() - timedelta(days=random.randint(0, 90), hours=random.randint(0,23), minutes=random.randint(0,59))
    return t.strftime("%Y-%m-%d %H:%M:%S")

def random_amount(bad_rate=0.07):
    if random.random() < bad_rate:
        return random.choice(["", "abc", "NaN"])
    # produce skewed distribution with outliers
    r = random.random()
    if r < 0.85:
        return round(random.gauss(500, 800), 2)  # everyday amounts (can be negative from gauss) - handle next
    if r < 0.98:
        return round(random.uniform(2000, 20000), 2)  # bigger purchases
    return round(random.uniform(50000, 200000), 2)  # extreme outlier

def random_user():
    # mix string and numeric-like IDs
    if random.random() < 0.2:
        return str(random.randint(1000, 9999))
    if random.random() < 0.1:
        return "user_" + uuid.uuid4().hex[:6]
    return random.choice(["U1001", "U1002", "U1003", "A101", "B202"])

def random_merchant():
    m = random.choice(merchants + ["Unknown", "unlisted", "flipkartk"])  # include typos
    # random casing
    return m if random.random() < 0.9 else m.upper()

def random_method():
    if random.random() < 0.05:
        return random.randint(0, 999)  # garbage numeric
    return random.choice(methods + ["wallet", None])

def random_country():
    if random.random() < 0.05:
        return "XX"  # invalid
    return random.choice(countries)

def generate_row(i):
    amt = random_amount()
    # correct negative/very small numbers to empty to mimic bad data
    if isinstance(amt, float) and amt < 1:
        amt = ""
    return {
        "timestamp": random_timestamp(bad_rate=0.08),
        "user_id": random_user(),
        "amount": amt,
        "merchant": random_merchant(),
        "method": random_method(),
        "country": random_country(),
        "extra_col": "x" if random.random() < 0.02 else ""
    }

def main():
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","user_id","amount","merchant","method","country","extra_col"])
        writer.writeheader()
        for i in range(N):
            row = generate_row(i)
            # produce some exact missing fields randomly
            if random.random() < 0.03:
                row["amount"] = ""
            if random.random() < 0.02:
                row["merchant"] = ""
            writer.writerow(row)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
