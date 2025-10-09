# addrdb.py
import sqlite3
import pandas as pd

print("🔍 Reading Book2.xlsx...")
df = pd.read_excel("Book2.xlsx")

# Ensure required columns exist
required_cols = ['postal', 'name', 'company', 'address', 'city', 'province', 'country', 'phone', 'email']
for col in required_cols:
    if col not in df.columns:
        print(f"⚠️ Warning: Column '{col}' not found in Excel. Using empty values.")
        df[col] = ""

# Generate postal_prefix if missing
if 'postal_prefix' not in df.columns:
    df['postal_prefix'] = df['postal'].astype(str).str[:3].str.upper()

# Keep only needed columns and clean
df = df[['name','company','phone','email','postal','postal_prefix','address','city','province','country']].copy()
df['postal_prefix'] = df['postal_prefix'].astype(str).str[:3].str.upper()
df.dropna(subset=['postal_prefix'], inplace=True)
df = df[df['postal_prefix'].str.len() == 3]

print(f"✅ Found {len(df)} valid addresses with postal_prefix")

# Save to SQLite
conn = sqlite3.connect("addrdb.db")
df.to_sql("addresses", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_prefix ON addresses(postal_prefix)")
conn.close()

print("✅ addrdb.db created successfully!")