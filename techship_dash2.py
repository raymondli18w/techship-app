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

        # ✅ Create INDIVIDUAL package object for EACH box (will be chunked later)
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
        chunk_info = f"({count // 50 + 1} API calls)" if count > 50 else ""
        st.write(f"📦 {order_id}: {count} box(es) {chunk_info}")
    
    return packages, carrier

# ✅ Submit single chunk (max 50 packages per API call)
def submit_chunk(payload, client_code, order_id, batch_id, dry_run=True, chunk_num=1, total_chunks=1):
    session = create_robust_session()
    timeout = 120  # 2 minutes per chunk
    
    # ✅ Default failed response structure (consistent keys)
    def failed_response(error_msg):
        return {
            "success": False,
            "cost": 0,
            "base_amount": 0,
            "fuel_surcharge": 0,
            "public_total": 0,
            "service": "N/A",
            "carrier": payload.get("CarrierCode", "N/A"),
            "error": error_msg,
            "boxes": len(payload.get("Packages", [])),
            "chunk_num": chunk_num,
            "total_chunks": total_chunks
        }
    
    try:
        payload["ClientCode"] = client_code
        params = {"dryRun": "true" if dry_run else "false"}
        
        response = session.post(API_URL, headers=HEADERS, json=payload, params=params, timeout=timeout)

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "No details"
            return failed_response(f"HTTP {response.status_code}: {error_text}")

        try:
            response_data = response.json()
            if not isinstance(response_data, dict):
                response_data = {}
        except Exception:
            return failed_response("Invalid JSON response")

        rates = response_data.get("Rates")
        if rates and isinstance(rates, list) and len(rates) > 0:
            best_rate = next((r for r in rates if r.get("IsBest")), rates[0])
            return {
                "success": True,
                "cost": best_rate.get("TotalAmount", best_rate.get("Amount", 0)),
                "base_amount": best_rate.get("BaseAmount", 0),
                "fuel_surcharge": best_rate.get("FuelSurcharge", 0),
                "public_total": best_rate.get("PublicTotalAmount", 0),
                "service": best_rate.get("ServiceName", best_rate.get("ServiceCode", "N/A")),
                "carrier": payload.get("CarrierCode", "N/A"),
                "error": None,
                "boxes": len(payload.get("Packages", [])),
                "chunk_num": chunk_num,
                "total_chunks": total_chunks
            }
        else:
            return failed_response("No rates returned")

    except requests.exceptions.Timeout:
        return failed_response(f"Timeout after {timeout}s")
    except Exception as e:
        return failed_response(str(e)[:150])
    finally:
        session.close()

# ✅ HYBRID: Chunk large orders, submit each chunk, sum results
def submit_all_shipments(packages, carrier, fallback_client_code, batch_id, dry_run=True, max_workers=2, chunk_size=50):
    """Submit packages in chunks for large orders, sum up costs"""
    actual_workers = min(max_workers, 4)
    
    # Group packages by OrderID
    orders = defaultdict(list)
    for pkg in packages:
        orders[pkg["OrderID"]].append(pkg)
    
    all_results = []
    
    for order_id, order_packages in orders.items():
        total_boxes = len(order_packages)
        num_chunks = (total_boxes + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Show progress for this order
        if num_chunks > 1:
            st.info(f"⏳ {order_id}: Processing {total_boxes} boxes in {num_chunks} chunks...")
        
        # Split into chunks
        chunks = []
        for i in range(0, total_boxes, chunk_size):
            chunk_packages = order_packages[i:i + chunk_size]
            chunks.append(chunk_packages)
        
        # Submit each chunk
        chunk_results = []
        transaction_number = str(uuid.uuid4()).replace("-", "")[:20]
        customer_order = order_id[:20]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = []
            for chunk_num, chunk_packages in enumerate(chunks, 1):
                # Build payload for this chunk
                packages_array = []
                for pkg in chunk_packages:
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
                    "TransactionNumber": f"{transaction_number}-{chunk_num:03d}",
                    "CustomerOrder": customer_order,
                    "BatchNumber": batch_id,
                    "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
                    "Routing": {
                        "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
                        "ServiceCode": chunk_packages[0]["ServiceLevel"],
                        "FreightPaymentTerms": "Prepaid"
                    },
                    "ShipToAddress": chunk_packages[0]["Address"],
                    "Packages": packages_array
                }
                
                client_code_val = chunk_packages[0].get("ClientCode") or fallback_client_code
                
                future = executor.submit(submit_chunk, payload, client_code_val, order_id, batch_id, dry_run, chunk_num, num_chunks)
                futures.append(future)
            
            # Collect chunk results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:  # ✅ Ensure result is not None
                    chunk_results.append(result)
        
        # ✅ Sum up all chunk results for this order (with safe defaults)
        total_cost = sum(r.get("cost", 0) for r in chunk_results if r.get("success", False))
        total_base = sum(r.get("base_amount", 0) for r in chunk_results if r.get("success", False))
        total_fuel = sum(r.get("fuel_surcharge", 0) for r in chunk_results if r.get("success", False))
        total_public = sum(r.get("public_total", 0) for r in chunk_results if r.get("success", False))
        successful_chunks = sum(1 for r in chunk_results if r.get("success", False))
        failed_chunks = num_chunks - successful_chunks
        
        # Get service info from first successful chunk
        service_info = "N/A"
        carrier_info = carrier
        error_info = None
        for r in chunk_results:
            if r.get("success"):
                service_info = r.get("service", "N/A")
                carrier_info = r.get("carrier", carrier)
                break
        
        if failed_chunks > 0:
            error_parts = []
            for r in chunk_results:
                if not r.get("success") and r.get("error"):
                    error_parts.append(f"Chunk {r.get('chunk_num', '?')}: {r.get('error', 'Unknown')}")
            if error_parts:
                error_info = "; ".join(error_parts[:3])  # Show first 3 errors
            if failed_chunks == num_chunks:
                error_info = f"All {num_chunks} chunks failed: {error_info}"
            else:
                error_info = f"{failed_chunks}/{num_chunks} chunks failed: {error_info}"
        
        status_text = "✅ Estimate (DryRun)" if dry_run else "✅ Saved to DB"
        if failed_chunks > 0 and successful_chunks > 0:
            status_text = f"⚠️ Partial ({successful_chunks}/{num_chunks} chunks)"
        if failed_chunks == num_chunks:
            status_text = "❌ Failed"
        
        all_results.append({
            "Status": status_text,
            "OrderID": order_id,
            "TransactionNumber": transaction_number,
            "BatchID": batch_id,
            "Boxes": total_boxes,
            "Cost": f"${total_cost:.2f}",
            "BaseAmount": f"${total_base:.2f}",
            "FuelSurcharge": f"${total_fuel:.2f}",
            "PublicTotal": f"${total_public:.2f}",
            "Service": service_info,
            "Carrier": carrier_info,
            "Recipient": order_packages[0]["Address"]["Name"],
            "PostalCode": order_packages[0]["Address"]["Postal"],
            "ExpectedDelivery": "N/A",
            "Zone": "N/A",
            "Chunks": f"{successful_chunks}/{num_chunks}",
            "Error": error_info,
            "DryRun": dry_run
        })
        
        if num_chunks > 1:
            if successful_chunks > 0:
                st.success(f"✅ {order_id}: {total_boxes} boxes processed in {num_chunks} chunks. Total: ${total_cost:.2f}")
            else:
                st.error(f"❌ {order_id}: All {num_chunks} chunks failed. Check errors below.")
    
    all_results.sort(key=lambda x: x.get("OrderID", ""))
    return all_results

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
    st.markdown("### Hybrid Chunking Mode — Large Orders Split Automatically")

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
        **✅ Hybrid Chunking**: Orders >50 boxes auto-split into 50-box chunks
        """)
        
        st.info("⏳ **Chunking Settings:**")
        st.markdown("""
        | Boxes | API Calls | Est. Time |
        |-------|-----------|-----------|
        | 1-50 | 1 call | ~30s |
        | 51-100 | 2 calls | ~60s |
        | 101-500 | 3-10 calls | ~5 min |
        | 501-2000 | 11-40 calls | ~15 min |
        """)
        
        chunk_size = st.slider("Chunk Size (boxes per API call)", 25, 100, 50, 
                               help="Smaller = more stable, Larger = fewer API calls")
        st.session_state.chunk_size = chunk_size
        
        if dry_run:
            st.warning("⚠️ **Dry Run Mode ON** — Estimates NOT saved to database.")
        else:
            st.success("✅ **Dry Run Mode OFF** — Estimates SAVED to database.")
        
        st.markdown("---")
        st.markdown("**🌐 TechSHIP Web Portal:**")
        st.code("https://18wheels.techship.ca/", language="text")
        st.markdown("Search by `BatchID` or `TransactionNumber` from results below.")

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
        
        chunk_size = st.session_state.get("chunk_size", 50)
        
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
                estimated_calls = (total_boxes + chunk_size - 1) // chunk_size
                estimated_time = estimated_calls * 30  # ~30s per API call
                st.warning(f"⏳ **Large Shipment:** {total_boxes} boxes = ~{estimated_calls} API calls. Estimated time: ~{estimated_time // 60} minutes. Please do not close this window.")

        st.subheader("⚙️ Configuration")
        max_workers = st.slider("Parallel Workers", 1, 4, 2, help="Lower workers = more stable for large orders")

        batch_id = str(uuid.uuid4()).replace("-", "")[:20]
        st.session_state.batch_id = batch_id

        with st.spinner(f"📤 Getting estimates for {total_boxes} packages (chunk size: {chunk_size})..."):
            all_results = submit_all_shipments(
                packages, carrier, fallback_client_code.strip(),
                batch_id=batch_id,
                dry_run=dry_run,
                max_workers=max_workers,
                chunk_size=chunk_size
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
                "Chunks": r.get("Chunks", "1/1"),
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
            if r.get("Error"):
                row["Error"] = r["Error"]
            display_data.append(row)
        
        results_df = pd.DataFrame(display_data)
        display_cols = [
            "OrderID", "Status", "BatchID", "TransactionNumber",
            "Boxes", "Chunks", "Cost", "BaseAmount", "FuelSurcharge", "PublicTotal", 
            "Service", "Carrier", "Recipient", "PostalCode"
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
