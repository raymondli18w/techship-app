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
# TechSHIP API Configuration (ESTIMATE ONLY - NO LABELS, NO CHARGES)
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
# Carrier Service Mapping — INCLUDING NEW CARRIERS
# =========================
CARRIER_SERVICE_MAP = {
    "FEDEX": {"CarrierCode": "FDXE", "Services": {"F1 - Priority Overnight": "F1", "F2 - Ground": "F2", "F3 - Express Saver": "F3"}},
    "PURO": {"CarrierCode": "PURO", "Services": {"P - Purolator Ground": "P", "PXPU - Purolator Express": "PXPU"}},
    "UPS": {"CarrierCode": "UPS", "Services": {"U - UPS Ground": "U", "EXP1 - UPS Express": "EXP1"}},
    "RS": {"CarrierCode": "RS", "Services": {"RateShopping": ""}},
    # ✅ NEW CARRIERS ADDED BELOW
    "UNI": {"CarrierCode": "UNIUNI", "Services": {"UNI - Standard": "UNI"}},
    "UBI": {"CarrierCode": "UBI", "Services": {"UBI - Intelcom Domestic": "UBI"}},
    "CANPAR": {"CarrierCode": "CNTL", "Services": {"CPR - Ground": "CPR"}}
}

# Rebuild SERVICE_TO_CARRIER mapping
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
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
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
        service_val = str(row['services']).strip() if pd.notna(row['services']) else ""
        
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
        def safe_string(col, default=""): return str(row[col]).strip() if pd.notna(row[col]) else default
        def safe_float(col, default=None):
            try: return float(row[col]) if pd.notna(row[col]) else default
            except (ValueError, TypeError): return default
        def safe_int(col, default=1):
            try: val = int(row[col]) if pd.notna(row[col]) else default; return max(1, val)
            except (ValueError, TypeError): return default

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
        service_level = row['resolved_service']
        
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
            st.info(f"📍 Auto-filled from DB: {address1}, {city}")
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
            if postal_prefix:
                st.warning(f"⚠️ Postal prefix '{postal_prefix}' not found in database")

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
            "Country": country,
            "Postal": postal.replace(" ", "").upper(),
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
            st.warning(f"⚠️ Row {int(idx) + 2}: No valid weights. Using default = 1.0")

        for dim in dimension_sets:
            for weight_col, weight_val in weight_values.items():
                for box_num in range(num_boxes):
                    unique_shipment_id = str(uuid.uuid4()).replace("-", "")[:16]
                    packages.append({
                        "SKU": safe_string('sku', "N/A"),
                        "Weight": weight_val,
                        "Description": safe_string('description', "No description"),
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
                        "OrderID": user_order_id
                    })

    if not packages:
        st.error("❌ No valid packages created.")
        return None
    return packages, carrier

def submit_single_shipment(payload, client_code, order_id, batch_id):
    session = create_robust_session()
    original_payload = payload.copy()
    try:
        payload["ClientCode"] = client_code
        response = session.post(API_URL, headers=HEADERS, json=payload, timeout=30)

        if response.status_code != 200:
            error_text = response.text[:200] if response.text else "No details"
            return {
                "Status": f"❌ HTTP {response.status_code}",
                "TransactionNumber": payload.get("TransactionNumber", "N/A"),
                "TrackingNumber": "N/A",
                "Cost": "$0.00",
                "Service": payload.get("Routing", {}).get("ServiceCode", "N/A"),
                "Recipient": payload.get("ShipToAddress", {}).get("Name", "N/A"),
                "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "N/A"),
                "Boxes": len(payload.get("Packages", [])),
                "Error": f"HTTP {response.status_code}: {error_text}",
                "_original_payload": original_payload,
                "ClientCode": client_code,
                "OrderID": order_id,
                "BatchID": batch_id
            }

        try:
            response_data = response.json()
            if not isinstance(response_data, dict):
                response_data = {}
        except Exception:
            return {
                "Status": "❌ Invalid JSON",
                "Error": "Response was not valid JSON",
                "TransactionNumber": payload.get("TransactionNumber", "N/A"),
                "TrackingNumber": "N/A",
                "Cost": "$0.00",
                "Service": payload.get("Routing", {}).get("ServiceCode", "N/A"),
                "Recipient": payload.get("ShipToAddress", {}).get("Name", "N/A"),
                "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "N/A"),
                "Boxes": len(payload.get("Packages", [])),
                "_original_payload": original_payload,
                "ClientCode": client_code,
                "OrderID": order_id,
                "BatchID": batch_id
            }

        rates = response_data.get("Rates")
        if rates and isinstance(rates, list) and len(rates) > 0:
            rate = rates[0]
            total_cost = rate.get("TotalCharge", 0)
            service_code = rate.get("ServiceCode", payload.get("Routing", {}).get("ServiceCode", "N/A"))
        else:
            total_cost = 0
            service_code = payload.get("Routing", {}).get("ServiceCode", "N/A")

        return {
            "Status": "✅ Success",
            "TransactionNumber": payload.get("TransactionNumber", "N/A"),
            "TrackingNumber": "N/A (Estimate)",
            "Cost": f"${total_cost:.2f}",
            "Service": service_code,
            "Recipient": payload["ShipToAddress"]["Name"],
            "PostalCode": payload["ShipToAddress"]["Postal"],
            "Boxes": len(payload["Packages"]),
            "LabelURL": "",
            "_original_payload": original_payload,
            "ClientCode": client_code,
            "OrderID": order_id,
            "BatchID": batch_id
        }

    except Exception as e:
        return {
            "Status": "❌ Failed",
            "TransactionNumber": payload.get("TransactionNumber", "unknown"),
            "TrackingNumber": "N/A",
            "Cost": "$0.00",
            "Service": payload.get("Routing", {}).get("ServiceCode", "unknown"),
            "Recipient": payload.get("ShipToAddress", {}).get("Name", "unknown"),
            "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "unknown"),
            "Boxes": len(payload.get("Packages", [])),
            "Error": str(e)[:150],
            "_original_payload": original_payload,
            "ClientCode": client_code,
            "OrderID": order_id,
            "BatchID": batch_id
        }
    finally:
        session.close()

# ✅ Accepts override_batch_id for shared batch
def submit_concurrent_shipments(packages, carrier, fallback_client_code, max_workers=3, override_batch_id=None):
    actual_workers = min(max_workers, 4)
    shipment_groups = defaultdict(list)
    for package in packages:
        key = (
            package["Address"]["Name"],
            package["Address"]["Address1"], 
            package["Address"]["City"],
            package["Address"]["Postal"],
            package["ServiceLevel"],
            package["Length"],
            package["Width"], 
            package["Height"],
            package["Weight"],
            package["PackagingWeight"]
        )
        shipment_groups[key].append(package)
    
    batch_id = override_batch_id if override_batch_id else str(uuid.uuid4()).replace("-", "")[:20]
    
    payloads = []
    for key, package_group in shipment_groups.items():
        customer_order = package_group[0].get("OrderID", str(uuid.uuid4()).replace("-", "")[:20])[:20]
        transaction_number = str(uuid.uuid4()).replace("-", "")[:20]
        
        payload = {
            "TransactionNumber": transaction_number,
            "CustomerOrder": customer_order,
            "BatchNumber": batch_id,
            "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
            "Routing": {
                "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
                "ServiceCode": package_group[0]["ServiceLevel"],
                "FreightPaymentTerms": "Prepaid"
            },
            "ShipToAddress": package_group[0]["Address"],
            "Packages": []
        }
        
        client_code_val = package_group[0].get("ClientCode") or fallback_client_code
        
        for pkg in package_group:
            payload["Packages"].append({
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
        payloads.append((payload, client_code_val, customer_order, batch_id))
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        for payload, client_code_val, order_id, bid in payloads:
            future = executor.submit(submit_single_shipment, payload, client_code_val, order_id, bid)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results, batch_id

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
    st.markdown("### First 3 ready fast — all orders share one batch")

    fallback_client_code = st.text_input("Fallback Client Code", value="omrtest1")
    if not fallback_client_code.strip():
        st.warning("⚠️ Please enter a valid Fallback Client Code")
        st.stop()

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
        """)
        st.info("✅ Example: carrier=UNI, services=UNI")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
    with col2:
        text_input = st.text_area("📋 Or Paste Data", height=150)

    # Initialize session state
    if "first_results" not in st.session_state:
        st.session_state.first_results = []
        st.session_state.background_results = []
        st.session_state.background_done = False
        st.session_state.batch_id = ""

    if st.button("🚀 Get Rate Estimates", type="primary"):
        with st.spinner("🔍 Parsing data..."):
            df = parse_input_data(text_input, uploaded_file)
            if df is None: st.stop()
            result = validate_and_process_data(df, fallback_client_code.strip(), force_rs=trigger_rs)
            if result is None: st.stop()
            packages, carrier = result
            st.success(f"✅ Parsed {len(packages)} packages for {carrier}")

        st.subheader("⚙️ Configuration")
        max_workers = st.slider("ParallelGroups", 1, 4, 3)

        # ✅ ONE batch_id for ALL
        batch_id = str(uuid.uuid4()).replace("-", "")[:20]
        st.session_state.batch_id = batch_id

        first_packages = packages[:3]
        rest_packages = packages[3:]

        st.session_state.first_results = []
        st.session_state.background_results = []
        st.session_state.background_done = False

        # Submit first 3
        if first_packages:
            with st.spinner(f"📤 Getting first {len(first_packages)} estimates..."):
                first_results, _ = submit_concurrent_shipments(
                    first_packages, carrier, fallback_client_code.strip(),
                    max_workers=1,
                    override_batch_id=batch_id
                )
                st.session_state.first_results = first_results

        # Submit rest in background
        if rest_packages:
            def background_task():
                results, _ = submit_concurrent_shipments(
                    rest_packages, carrier, fallback_client_code.strip(),
                    max_workers,
                    override_batch_id=batch_id
                )
                st.session_state.background_results = results
                st.session_state.background_done = True

            thread = threading.Thread(target=background_task, daemon=True)
            thread.start()
            st.info(f"⏳ {len(rest_packages)} orders in background. BatchID: `{batch_id[:8]}...`")
        else:
            st.session_state.background_done = True

    # Display results
    all_results = st.session_state.first_results + st.session_state.background_results
    if all_results:
        success_count = sum(1 for r in all_results if "Success" in r.get("Status", ""))
        failed_count = len(all_results) - success_count

        st.subheader("📊 Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(all_results))
        col2.metric("Success", success_count)
        col3.metric("Failed", failed_count)

        display_data = []
        for r in all_results:
            row = {
                "Status": r.get("Status", "Unknown"),
                "OrderID": r.get("OrderID", "N/A"),
                "BatchID": r.get("BatchID", "N/A"),
                "TransactionNumber": r.get("TransactionNumber", "N/A"),
                "Boxes": r.get("Boxes", 0),
                "TrackingNumber": r.get("TrackingNumber", ""),
                "Cost": r.get("Cost", "$0.00"),
                "Service": r.get("Service", ""),
                "Recipient": r.get("Recipient", ""),
                "PostalCode": r.get("PostalCode", "")
            }
            if "Error" in r:
                row["Error"] = r["Error"]
            display_data.append(row)
        
        results_df = pd.DataFrame(display_data)
        display_cols = [
            "Status", "OrderID", "BatchID", "TransactionNumber",
            "Boxes", "TrackingNumber", "Cost", "Service", "Recipient", "PostalCode"
        ]
        if "Error" in results_df.columns:
            display_cols.append("Error")
        st.dataframe(results_df[display_cols], use_container_width=True)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download All Results",
            csv,
            f"techship_estimates_{st.session_state.batch_id}.csv",
            "text/csv"
        )

    # Auto-refresh or manual check
    if not st.session_state.background_done and st.session_state.background_results:
        time.sleep(1)
        st.rerun()

    if not st.session_state.background_done and len(st.session_state.first_results) > 0:
        if st.button("🔄 Check for New Results"):
            st.rerun()

if __name__ == "__main__":
    main()