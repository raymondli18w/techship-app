import sqlite3
import csv

# Create/connect database
conn = sqlite3.connect("canada_addresses.db")
cur = conn.cursor()

# Create table
cur.execute("""
CREATE TABLE IF NOT EXISTS addresses (
    id INTEGER PRIMARY KEY,
    postal_code TEXT NOT NULL,
    address TEXT,
    city TEXT,
    province TEXT,
    country TEXT
)
""")

# Optional: create index for fast first-3-characters lookup
cur.execute("CREATE INDEX IF NOT EXISTS idx_postal_prefix ON addresses(substr(postal_code,1,3))")

# Example: bulk insert from CSV
with open("canada_addresses.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data = [(row['postal'], row['address'], row['city'], row['province'], row['country']) for row in reader]

cur.executemany("INSERT INTO addresses (postal_code, address, city, province, country) VALUES (?, ?, ?, ?, ?)", data)
conn.commit()
conn.close()
