import streamlit as st
import pandas as pd
import requests
import uuid
import concurrent.futures
import threading
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO, BytesIO
from collections import defaultdict

# SQLite Address Lookup
from sqlite_lookup import get_address_by_prefix

# =========================
# MUST be the first Streamlit command
# =========================
st.set_page_config(page_title="TechSHIP Bulk Rate Estimator", page_icon="📦", layout="wide")

# =========================
# TechSHIP API Configuration (ESTIMATE API)
# =========================
API_URL = "https://18wheels.techship.ca/api/v3/shipments/estimate"
API_KEY = "bfdcbf84-f76d-b85b-8eae-fa925d6fa863"
API_SECRET = "d2caf6ab27688a76966f1b8b6cbc2029"
HEADERS = {
    "x-api-key": API_KEY,
    "x-secret-key": API_SECRET,
    "Content-Type": "application/json"
}

# =========================
# Carrier Service Mapping
# =========================
CARRIER_SERVICE_MAP = {
    "FEDEX": {"CarrierCode": "FDXE", "Services": {"F1 - Priority Overnight": "F1", "F2 - Ground": "F2", "F3 - Express Saver": "F3"}},
    "PURO": {"CarrierCode": "PURO", "Services": {"P - Purolator Ground": "P", "PXPU - Purolator Express": "PXPU"}},
    "UPS": {"CarrierCode": "UPS", "Services": {"U - UPS Ground": "U", "EXP1 - UPS Express": "EXP1"}},
    "RS": {"CarrierCode": "RS", "Services": {"RateShopping": ""}},
    "UNI": {"CarrierCode": "UNIUNI", "Services": {"UNI - Standard": "UNI"}},
    "UBI": {"CarrierCode": "UBI", "Services": {"UBI - Intelcom Domestic": "UBI"}},
    "CANPAR": {"CarrierCode": "CNTL", "Services": {"CPR - Ground": "CPR"}}
}

SERVICE_TO_CARRIER = {}
for carrier, info in CARRIER_SERVICE_MAP.items():
    for service_name, service_code in info["Services"].items():
        if service_code:
            SERVICE_TO_CARRIER[service_code] = carrier

# =========================
# Helper Functions
# =========================
def create_robust_session():
    """Create session with extended timeouts and connection pooling"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=200,
        pool_maxsize=200,
        pool_block=False
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def parse_input_data(text_data, uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(BytesIO(uploaded_file.getvalue()))
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(BytesIO(uploaded_file.getvalue()))
            else:
                st.error("❌ Unsupported file type. Please use CSV or Excel.")
                return None
        except Exception as e:
            st.error(f"❌ File parsing error: {str(e)}")
            return None
    else:
        if not text_data or not text_data.strip():
            st.error("❌ Please provide data via text or file upload.")
            return None
        delimiter = '\t' if '\t' in text_data[:500] else ','
        try:
            return pd.read_csv(StringIO(text_data), delimiter=delimiter, skipinitialspace=True)
        except Exception as e:
            st.error(f"❌ Data parsing error: {str(e)}")
            return None

def validate_and_process_data(df, fallback_client_code, force_rs=False):
    clean_columns = []
    for col in df.columns:
        cleaned = str(col).strip().lower()
        cleaned = cleaned.replace(' ', '_').replace('-', '_').replace('.', '_')
        cleaned = '_'.join([part for part in cleaned.split('_') if part])
        clean_columns.append(cleaned)
    df.columns = clean_columns

    # ✅ Drop any empty column names (caused by trailing commas in CSV)
    df = df.loc[:, df.columns != '']

    column_mapping = {
        'Services': 'services', 'service': 'services', 'service_code': 'services',
        'address1': 'address', 'street': 'address', 'street1': 'address',
        'address2': 'address2', 'street2': 'address2', 'suite': 'address2',
        'state': 'province', 'zip': 'postal', 'zipcode': 'postal',
        'item_sku': 'sku', 'product_sku': 'sku',
        'item_description': 'description', 'desc': 'description',
        'pkg_weight': 'packaging_weight', 'num_boxes': 'boxes',
        'contact_name': 'name', 'phone_number': 'phone', 'email_address': 'email',
        'postalzip2': 'postal_prefix',
        'clientcode': 'client_code',
        'client_code': 'client_code',
        'order_id': 'order_id',
        'orderid': 'order_id',
        'purchase_order': 'order_id',
        'po_number': 'order_id',
        'carrier': 'carrier',
        'carreir': 'carrier',
        'carrier_code': 'carrier',
        'length': 'length',
        'width': 'width',
        'height': 'height'
    }
    df.columns = [column_mapping.get(col, col) for col in df.columns]

    essential_columns = ['name', 'services']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None

    lwh_columns = [col for col in df.columns if col.startswith('lwh')]
    has_manual_dims = all(col in df.columns for col in ['length', 'width', 'height'])

    if not lwh_columns and not has_manual_dims:
        st.error("❌ Either provide 'lwh', 'lwh2', ... columns OR 'length', 'width', 'height' columns.")
        return None

    weight_columns = [col for col in df.columns if col.startswith('weight')]
    if not weight_columns:
        weight_columns = ['weight']

    carrier_col = df.get('carrier', pd.Series([''] * len(df))).astype(str).str.strip().str.upper()
    if force_rs:
        carrier_col = pd.Series(['RS'] * len(df))
    
    detected_carriers = set()
    service_levels = []
    
    for idx, row in df.iterrows():
        carrier_val = carrier_col.iloc[idx] if idx < len(carrier_col) else ""
        service_val = str(row.get('services')).strip() if pd.notna(row.get('services')) else ""
        
        if carrier_val == "RS" or service_val == "":
            detected_carriers.add("RS")
            service_levels.append("")
        elif carrier_val in CARRIER_SERVICE_MAP:
            if service_val in SERVICE_TO_CARRIER and SERVICE_TO_CARRIER[service_val] == carrier_val:
                detected_carriers.add(carrier_val)
                service_levels.append(service_val)
            else:
                valid_services = [k for k, v in SERVICE_TO_CARRIER.items() if v == carrier_val]
                st.error(f"❌ Row {int(idx) + 2}: Invalid service '{service_val}' for {carrier_val}. Valid: {valid_services}")
                return None
        else:
            if service_val in SERVICE_TO_CARRIER:
                carrier_from_service = SERVICE_TO_CARRIER[service_val]
                detected_carriers.add(carrier_from_service)
                service_levels.append(service_val)
            elif service_val == "":
                detected_carriers.add("RS")
                service_levels.append("")
            else:
                valid_codes = list(SERVICE_TO_CARRIER.keys()) + ["(leave blank for RS)"]
                st.error(f"❌ Row {int(idx) + 2}: Unknown service code '{service_val}'. Valid codes: {valid_codes}")
                return None
    
    df['resolved_service'] = service_levels

    if len(detected_carriers) == 0:
        st.error("❌ No valid service codes found. Use codes like: F2, P, U, UNI, UBI, CPR, or leave empty for RS")
        return None
    if len(detected_carriers) > 1:
        st.error(f"❌ Mixed carriers detected: {list(detected_carriers)}. Use one carrier per batch.")
        return None

    carrier = list(detected_carriers)[0]
    packages = []

    for idx, row in df.iterrows():
        def safe_string(col, default=""): 
            val = row.get(col)
            if pd.isna(val):
                return default
            return str(val).strip()
        
        def safe_float(col, default=None):
            val = row.get(col)
            if pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        def safe_int(col, default=1):
            val = row.get(col)
            if pd.isna(val):
                return default
            try:
                return max(1, int(val))
            except (ValueError, TypeError):
                return default

        user_name = safe_string('name')
        user_company = safe_string('company')
        user_address1 = safe_string('address')
        user_address2 = safe_string('address2')
        user_city = safe_string('city')
        user_province = safe_string('province')
        user_country = safe_string('country')
        user_postal = safe_string('postal')
        user_phone = safe_string('phone')
        user_email = safe_string('email')
        service_level = row.get('resolved_service', '')
        
        user_order_id = safe_string('order_id')
        if not user_order_id:
            user_order_id = str(uuid.uuid4()).replace("-", "")[:20]
        
        packaging_weight = safe_float('packaging_weight', 0.0)
        num_boxes = safe_int('boxes', 1)
        
        # ✅ WARNING: Large box counts will take longer but will NOT be split
        if num_boxes > 100:
            estimated_time = num_boxes * 0.5  # ~0.5 sec per box
            st.warning(f"⏳ Row {int(idx) + 2}: **{num_boxes} boxes** detected. Estimated processing time: ~{estimated_time:.0f} seconds. This is normal — please wait.")
        if num_boxes > 1000:
            st.warning(f"⚠️ Row {int(idx) + 2}: **{num_boxes} boxes** is a very large shipment. Processing may take several minutes.")
        
        row_client_code = safe_string('client_code') or fallback_client_code

        postal_prefix = safe_string('postal_prefix')
        db_entry = None
        if postal_prefix and len(postal_prefix) >= 3:
            db_entry = get_address_by_prefix(postal_prefix)

        if db_entry:
            address1 = user_address1 or db_entry["address"]
            city = user_city or db_entry["city"]
            province = user_province or db_entry["province"]
            postal = user_postal or db_entry["postal"]
            country = user_country or db_entry["country"]
            name = user_name or db_entry["name"]
            company = user_company or db_entry["company"]
            phone = user_phone or db_entry["phone"]
            email = user_email or db_entry["email"]
        else:
            address1 = user_address1
            city = user_city
            province = user_province
            postal = user_postal
            country = user_country or "Canada"
            name = user_name
            company = user_company
            phone = user_phone
            email = user_email

        if not name or not address1 or not city or not postal:
            st.error(f"❌ Row {int(idx) + 2}: Missing required field")
            return None

        address = {
            "Name": name,
            "Company": company,
            "Address1": address1,
            "Address2": user_address2,
            "City": city,
            "StateProvince": province,
            "Postal": postal.replace(" ", "").upper(),
            "Country": country,
            "Phone": phone,
            "Email": email
        }

        dimension_sets = []
        if lwh_columns:
            for lwh_col in lwh_columns:
                val = safe_float(lwh_col)
                if val is not None and val > 0:
                    dimension_sets.append({
                        "source": lwh_col,
                        "length": val,
                        "width": val,
                        "height": val
                    })
        elif has_manual_dims:
            length = safe_float('length')
            width = safe_float('width')
            height = safe_float('height')
            if length and width and height and length > 0 and width > 0 and height > 0:
                dimension_sets.append({
                    "source": "manual",
                    "length": length,
                    "width": width,
                    "height": height
                })
            else:
                st.error(f"❌ Row {int(idx) + 2}: Missing or invalid 'length', 'width', or 'height'")
                return None
        else:
            st.error(f"❌ Row {int(idx) + 2}: No valid dimensions found")
            return None

        weight_values = {}
        for weight_col in weight_columns:
            val = safe_float(weight_col)
            if val is not None and val > 0:
                weight_values[weight_col] = val
        
        if not weight_values:
            weight_values = {"weight": 1.0}

        # ✅ Create INDIVIDUAL package object for EACH box (no splitting)
        for dim in dimension_sets:
            for weight_col, weight_val in weight_values.items():
                for box_num in range(num_boxes):
                    unique_shipment_id = str(uuid.uuid4()).replace("-", "")[:16]
                    packages.append({
                        "SKU": safe_string('sku', "N/A"),
                        "Weight": weight_val,
                        "Description": f"{safe_string('description', 'No description')} - Box {box_num + 1}/{num_boxes}",
                        "Length": dim["length"],
                        "Width": dim["width"],
                        "Height": dim["height"],
                        "PackagingWeight": packaging_weight,
                        "Address": address,
                        "ServiceLevel": service_level,
                        "Carrier": carrier,
                        "ClientCode": row_client_code,
                        "LWH_Source": dim["source"],
                        "Weight_Source": weight_col,
                        "Box_Index": box_num,
                        "UNIQUE_SHIPMENT_ID": unique_shipment_id,
                        "OrderID": user_order_id,
                        "Row_Index": idx
                    })

    if not packages:
        st.error("❌ No valid packages created.")
        return None
    
    # Show package summary
    order_counts = defaultdict(int)
    for pkg in packages:
        order_counts[pkg["OrderID"]] += 1
    
    st.info(f"ℹ️ Created {len(packages)} individual package objects from {len(df)} CSV rows")
    for order_id, count in sorted(order_counts.items()):
        st.write(f"📦 {order_id}: {count} box(es)")
    
    return packages, carrier

# ✅ UPDATED: Extended timeout with retry logic for large orders
def submit_single_shipment(payload, client_code, order_id, batch_id, dry_run=True, num_boxes=1):
    session = create_robust_session()
    original_payload = payload.copy()
    
    # ✅ Dynamic timeout based on package count (NO upper limit)
    if num_boxes <= 50:
        timeout = 60
    elif num_boxes <= 100:
        timeout = 120
    elif num_boxes <= 500:
        timeout = 300  # 5 minutes
    elif num_boxes <= 1000:
        timeout = 600  # 10 minutes
    else:
        timeout = 900  # 15 minutes for 2000+ boxes
    
    try:
        payload["ClientCode"] = client_code
        params = {"dryRun": "true" if dry_run else "false"}
        
        # Show timeout info for large orders
        if num_boxes > 100:
            st.info(f"⏳ Processing {num_boxes} boxes... Timeout set to {timeout}s. Please wait.")
        
        response = session.post(API_URL, headers=HEADERS, json=payload, params=params, timeout=timeout)

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "No details"
            return {
                "Status": f"❌ HTTP {response.status_code}",
                "OrderID": order_id,
                "TransactionNumber": payload.get("TransactionNumber", "N/A"),
                "TrackingNumber": "N/A",
                "Cost": "$0.00",
                "BaseAmount": "$0.00",
                "FuelSurcharge": "$0.00",
                "Service": payload.get("Routing", {}).get("ServiceCode", "N/A"),
                "Carrier": payload.get("CarrierCode", "N/A"),
                "Recipient": payload.get("ShipToAddress", {}).get("Name", "N/A"),
                "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "N/A"),
                "Boxes": num_boxes,
                "ExpectedDelivery": "N/A",
                "Zone": "N/A",
                "Error": f"HTTP {response.status_code}: {error_text}",
                "ClientCode": client_code,
                "BatchID": batch_id,
                "DryRun": dry_run
            }

        try:
            response_data = response.json()
            if not isinstance(response_data, dict):
                response_data = {}
        except Exception:
            return {
                "Status": "❌ Invalid JSON",
                "Error": "Response was not valid JSON",
                "OrderID": order_id,
                "TransactionNumber": payload.get("TransactionNumber", "N/A"),
                "TrackingNumber": "N/A",
                "Cost": "$0.00",
                "BaseAmount": "$0.00",
                "FuelSurcharge": "$0.00",
                "Service": payload.get("Routing", {}).get("ServiceCode", "N/A"),
                "Carrier": payload.get("CarrierCode", "N/A"),
                "Recipient": payload.get("ShipToAddress", {}).get("Name", "N/A"),
                "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "N/A"),
                "Boxes": num_boxes,
                "ExpectedDelivery": "N/A",
                "Zone": "N/A",
                "ClientCode": client_code,
                "BatchID": batch_id,
                "DryRun": dry_run
            }

        rates = response_data.get("Rates")
        if rates and isinstance(rates, list) and len(rates) > 0:
            best_rate = next((r for r in rates if r.get("IsBest")), rates[0])
            total_cost = best_rate.get("TotalAmount", best_rate.get("Amount", 0))
            service_code = best_rate.get("ServiceCode", payload.get("Routing", {}).get("ServiceCode", "N/A"))
            service_name = best_rate.get("ServiceName", service_code)
            expected_delivery = best_rate.get("ExpectedDeliveryDate", "N/A")
            zone = best_rate.get("Zone", "N/A")
            fuel_surcharge = best_rate.get("FuelSurcharge", 0)
            base_amount = best_rate.get("BaseAmount", 0)
            public_total = best_rate.get("PublicTotalAmount", total_cost)
        else:
            total_cost = 0
            service_code = payload.get("Routing", {}).get("ServiceCode", "N/A")
            service_name = service_code
            expected_delivery = "N/A"
            zone = "N/A"
            fuel_surcharge = 0
            base_amount = 0
            public_total = 0

        status_text = "✅ Estimate (DryRun)" if dry_run else "✅ Saved to DB"

        return {
            "Status": status_text,
            "OrderID": order_id,
            "TransactionNumber": payload.get("TransactionNumber", "N/A"),
            "TrackingNumber": "N/A (Estimate Only)",
            "Cost": f"${total_cost:.2f}",
            "BaseAmount": f"${base_amount:.2f}",
            "FuelSurcharge": f"${fuel_surcharge:.2f}",
            "PublicTotal": f"${public_total:.2f}",
            "Service": f"{service_name} ({service_code})",
            "Carrier": payload.get("CarrierCode", "N/A"),
            "Recipient": payload["ShipToAddress"]["Name"],
            "PostalCode": payload["ShipToAddress"]["Postal"],
            "Boxes": num_boxes,
            "ExpectedDelivery": expected_delivery,
            "Zone": zone,
            "ClientCode": client_code,
            "BatchID": batch_id,
            "DryRun": dry_run
        }

    except requests.exceptions.Timeout:
        return {
            "Status": "❌ Timeout",
            "OrderID": order_id,
            "TransactionNumber": payload.get("TransactionNumber", "unknown"),
            "TrackingNumber": "N/A",
            "Cost": "$0.00",
            "BaseAmount": "$0.00",
            "FuelSurcharge": "$0.00",
            "PublicTotal": "$0.00",
            "Service": payload.get("Routing", {}).get("ServiceCode", "unknown"),
            "Carrier": payload.get("CarrierCode", "unknown"),
            "Recipient": payload.get("ShipToAddress", {}).get("Name", "unknown"),
            "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "unknown"),
            "Boxes": num_boxes,
            "ExpectedDelivery": "N/A",
            "Zone": "N/A",
            "Error": f"Request timed out after {timeout}s. The API took too long. Try again or contact support.",
            "ClientCode": client_code,
            "BatchID": batch_id,
            "DryRun": dry_run
        }
    except Exception as e:
        return {
            "Status": "❌ Failed",
            "OrderID": order_id,
            "TransactionNumber": payload.get("TransactionNumber", "unknown"),
            "TrackingNumber": "N/A",
            "Cost": "$0.00",
            "BaseAmount": "$0.00",
            "FuelSurcharge": "$0.00",
            "PublicTotal": "$0.00",
            "Service": payload.get("Routing", {}).get("ServiceCode", "unknown"),
            "Carrier": payload.get("CarrierCode", "unknown"),
            "Recipient": payload.get("ShipToAddress", {}).get("Name", "unknown"),
            "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "unknown"),
            "Boxes": num_boxes,
            "ExpectedDelivery": "N/A",
            "Zone": "N/A",
            "Error": str(e)[:150],
            "ClientCode": client_code,
            "BatchID": batch_id,
            "DryRun": dry_run
        }
    finally:
        session.close()

def submit_all_shipments(packages, carrier, fallback_client_code, batch_id, dry_run=True, max_workers=4):
    """Submit packages grouped by OrderID - each order gets ALL its boxes in one API call"""
    actual_workers = min(max_workers, 4)  # Limit workers for large orders
    
    # Group packages by OrderID
    orders = defaultdict(list)
    for pkg in packages:
        orders[pkg["OrderID"]].append(pkg)
    
    payloads = []
    for order_id, order_packages in orders.items():
        customer_order = order_id[:20]
        transaction_number = str(uuid.uuid4()).replace("-", "")[:20]
        
        # Build Packages array with ALL individual boxes for this order
        packages_array = []
        for pkg in order_packages:
            packages_array.append({
                "Weight": pkg["Weight"],
                "Dimensions": {
                    "Length": pkg["Length"],
                    "Width": pkg["Width"],
                    "Height": pkg["Height"],
                    "PackagingWeight": pkg["PackagingWeight"]
                },
                "Items": [{
                    "SKU": pkg["SKU"],
                    "Description": pkg["Description"],
                    "Quantity": 1
                }]
            })
        
        payload = {
            "TransactionNumber": transaction_number,
            "CustomerOrder": customer_order,
            "BatchNumber": batch_id,
            "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
            "Routing": {
                "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
                "ServiceCode": order_packages[0]["ServiceLevel"],
                "FreightPaymentTerms": "Prepaid"
            },
            "ShipToAddress": order_packages[0]["Address"],
            "Packages": packages_array
        }
        
        client_code_val = order_packages[0].get("ClientCode") or fallback_client_code
        num_boxes = len(order_packages)
        
        payloads.append((payload, client_code_val, customer_order, batch_id, num_boxes))
    
    results = []
    total_orders = len(payloads)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(submit_single_shipment, payload, client_code_val, order_id, bid, dry_run, num_boxes) 
                   for payload, client_code_val, order_id, bid, num_boxes in payloads]
        
        # Show progress with order count
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            progress = (i + 1) / total_orders
            progress_bar.progress(progress)
            
            # Show which order is processing
            completed_orders = [r.get("OrderID", "?") for r in results]
            status_text.text(f"⏳ Processing: {i + 1}/{total_orders} orders completed")
        
        progress_bar.empty()
        status_text.empty()
    
    results.sort(key=lambda x: x.get("OrderID", ""))
    return results

def add_selectable_css():
    st.markdown("""
    <style>
    * {-webkit-user-select: text !important; -moz-user-select: text !important; user-select: text !important;}
    .stDataFrame, [data-testid="stMetricValue"], .stMarkdown, textarea {
        -webkit-user-select: text !important; -moz-user-select: text !important; user-select: text !important;
    }
    </style>""", unsafe_allow_html=True)

# =========================
# Streamlit Application
# =========================
def main():
    add_selectable_css()
    st.title("📦 TechSHIP Bulk Rate Estimator")
    st.markdown("### Extended Timeout Mode — Large Orders Supported (2000+ Boxes)")

    fallback_client_code = st.text_input("Fallback Client Code", value="omrtest1")
    if not fallback_client_code.strip():
        st.warning("⚠️ Please enter a valid Fallback Client Code")
        st.stop()

    dry_run = st.checkbox("🔒 Dry Run Mode (Estimates Only - Not Saved to DB)", value=True)

    col1, col2 = st.columns(2)
    with col1:
        trigger_rs = st.button("🎯 Trigger RS (RateShopping)", type="secondary")

    with st.sidebar:
        st.header("📋 Usage Guide")
        st.markdown("""
        **Required Columns**: `name`, `services`  
        **Carrier Options**:
        - `FEDEX` → services: `F1`, `F2`, `F3`
        - `PURO` → services: `P`, `PXPU`
        - `UPS` → services: `U`, `EXP1`
        - `UNI` → service: `UNI`
        - `UBI` → service: `UBI`
        - `CANPAR` → service: `CPR`
        - Leave `services` blank → RateShopping (RS)
        **Dimensions**: `lwh` or `length/width/height`  
        **Weights**: `weight`, `weight2`, ...
        **✅ Large Orders**: 2000+ boxes supported with extended timeouts
        """)
        
        st.info("⏳ **Timeout Settings:**")
        st.markdown("""
        | Boxes | Timeout |
        |-------|---------|
        | 1-50 | 60s |
        | 51-100 | 120s |
        | 101-500 | 300s (5 min) |
        | 501-1000 | 600s (10 min) |
        | 1000+ | 900s (15 min) |
        """)
        
        if dry_run:
            st.warning("⚠️ **Dry Run Mode ON** — Estimates NOT saved to database.")
        else:
            st.success("✅ **Dry Run Mode OFF** — Estimates SAVED to database.")
        
        st.markdown("---")
        st.markdown("**🌐 TechSHIP Web Portal:**")
        st.code("https://18wheels.techship.ca/", language="text")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
    with col2:
        text_input = st.text_area("📋 Or Paste Data", height=150)

    if "all_results" not in st.session_state:
        st.session_state.all_results = []
        st.session_state.batch_id = ""
        st.session_state.total_orders = 0
        st.session_state.processing_done = True

    if st.button("🚀 Get Rate Estimates", type="primary"):
        st.session_state.processing_done = False
        st.session_state.all_results = []
        
        with st.spinner("🔍 Parsing data..."):
            df = parse_input_data(text_input, uploaded_file)
            if df is None: 
                st.session_state.processing_done = True
                st.stop()
            result = validate_and_process_data(df, fallback_client_code.strip(), force_rs=trigger_rs)
            if result is None: 
                st.session_state.processing_done = True
                st.stop()
            packages, carrier = result
            st.session_state.total_orders = len(set(pkg["OrderID"] for pkg in packages))
            total_boxes = len(packages)
            st.success(f"✅ Parsed {st.session_state.total_orders} unique orders ({total_boxes} total packages)")
            
            if total_boxes > 500:
                st.warning(f"⏳ **Large Shipment:** {total_boxes} boxes detected. Processing may take 5-15 minutes. Please do not close this window.")

        st.subheader("⚙️ Configuration")
        max_workers = st.slider("Parallel Workers", 1, 4, 2, help="Lower workers = more stable for large orders")

        batch_id = str(uuid.uuid4()).replace("-", "")[:20]
        st.session_state.batch_id = batch_id

        with st.spinner(f"📤 Getting estimates for {total_boxes} packages..."):
            all_results = submit_all_shipments(
                packages, carrier, fallback_client_code.strip(),
                batch_id=batch_id,
                dry_run=dry_run,
                max_workers=max_workers
            )
            st.session_state.all_results = all_results
            st.session_state.processing_done = True

        st.success(f"✅ Completed! Processed {len(all_results)} orders")

    # Display results
    if st.session_state.all_results:
        all_results = st.session_state.all_results
        success_count = sum(1 for r in all_results if "✅" in r.get("Status", ""))
        failed_count = len(all_results) - success_count

        st.subheader("📊 Results - All Orders")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Orders", st.session_state.total_orders)
        col2.metric("Success", success_count)
        col3.metric("Failed", failed_count)

        display_data = []
        for r in all_results:
            row = {
                "OrderID": r.get("OrderID", "N/A"),
                "Status": r.get("Status", "Unknown"),
                "BatchID": r.get("BatchID", "N/A"),
                "TransactionNumber": r.get("TransactionNumber", "N/A"),
                "Boxes": r.get("Boxes", 0),
                "Cost": r.get("Cost", "$0.00"),
                "BaseAmount": r.get("BaseAmount", ""),
                "FuelSurcharge": r.get("FuelSurcharge", ""),
                "PublicTotal": r.get("PublicTotal", ""),
                "Service": r.get("Service", ""),
                "Carrier": r.get("Carrier", ""),
                "Recipient": r.get("Recipient", ""),
                "PostalCode": r.get("PostalCode", ""),
                "ExpectedDelivery": r.get("ExpectedDelivery", "N/A"),
                "Zone": r.get("Zone", "N/A")
            }
            if "Error" in r:
                row["Error"] = r["Error"]
            display_data.append(row)
        
        results_df = pd.DataFrame(display_data)
        display_cols = [
            "OrderID", "Status", "BatchID", "TransactionNumber",
            "Boxes", "Cost", "BaseAmount", "FuelSurcharge", "PublicTotal", 
            "Service", "Carrier", "Recipient", "PostalCode", "ExpectedDelivery", "Zone"
        ]
        if "Error" in results_df.columns:
            display_cols.append("Error")
        st.dataframe(results_df[display_cols], use_container_width=True)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download All Estimates",
            csv,
            f"techship_estimates_{st.session_state.batch_id}.csv",
            "text/csv"
        )

        if not dry_run:
            st.success(f"✅ **Saved to Database!** View in TechSHIP: [https://18wheels.techship.ca/](https://18wheels.techship.ca/) — Search BatchID: `{st.session_state.batch_id}`")
        else:
            st.info(f"ℹ️ **Dry Run Mode** — Not saved to database. BatchID: `{st.session_state.batch_id}`")

if __name__ == "__main__":
    main()
