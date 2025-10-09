# sqlite_lookup.py
import sqlite3

def get_address_by_prefix(prefix):
    """Look up address by 3-letter postal prefix. Returns dict or None."""
    if not prefix or len(prefix) < 3:
        return None
    prefix = prefix[:3].upper()
    
    try:
        conn = sqlite3.connect("addrdb.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT name, company, phone, email, address, city, province, country, postal
            FROM addresses
            WHERE postal_prefix = ?
            LIMIT 1
        """, (prefix,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            return {
                "name": row[0] or "",
                "company": row[1] or "",
                "phone": row[2] or "",
                "email": row[3] or "",
                "address": row[4] or "",
                "city": row[5] or "",
                "province": row[6] or "",
                "country": row[7] or "Canada",
                "postal": row[8] or ""
            }
    except Exception as e:
        print(f"❌ SQLite lookup error: {e}")
    return None