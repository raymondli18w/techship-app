import streamlit as st
import pandas as pd
import requests
import uuid
import time
import concurrent.futures
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import StringIO, BytesIO
from collections import defaultdict

# SQLite Address Lookup
from sqlite_lookup import get_address_by_prefix

# =========================
# TechSHIP API Configuration
# =========================
API_URL = "https://18wheels.techship.ca/api/v3/shipments/create?duplicateHandling=2"
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
    "RS": {"CarrierCode": "RS", "Services": {"RateShopping": ""}}
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
        'carreir': 'carrier',  # Fix common typo
        'carrier_code': 'carrier'
    }
    df.columns = [column_mapping.get(col, col) for col in df.columns]

    essential_columns = ['name', 'services']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None

    lwh_columns = [col for col in df.columns if col.startswith('lwh')]
    if not lwh_columns:
        st.error("❌ No 'lwh', 'lwh2', 'lwh3', etc. columns found. At least one is required.")
        return None

    # Detect weight columns
    weight_columns = [col for col in df.columns if col.startswith('weight')]
    if not weight_columns:
        weight_columns = ['weight']  # Fallback

    # Handle carrier column and service validation
    carrier_col = df.get('carrier', pd.Series([''] * len(df))).astype(str).str.strip().str.upper()
    
    # If force_rs button is clicked, override to RS
    if force_rs:
        carrier_col = pd.Series(['RS'] * len(df))
    
    detected_carriers = set()
    service_levels = []
    
    for idx, row in df.iterrows():
        carrier_val = carrier_col.iloc[idx] if idx < len(carrier_col) else ""
        service_val = str(row['services']).strip() if pd.notna(row['services']) else ""
        
        if carrier_val == "RS" or service_val == "":
            # Use RS carrier
            detected_carriers.add("RS")
            service_levels.append("")
        elif carrier_val in CARRIER_SERVICE_MAP:
            # Validate service code for this carrier
            if service_val in SERVICE_TO_CARRIER and SERVICE_TO_CARRIER[service_val] == carrier_val:
                detected_carriers.add(carrier_val)
                service_levels.append(service_val)
            else:
                valid_services = [k for k, v in SERVICE_TO_CARRIER.items() if v == carrier_val]
                st.error(f"❌ Row {int(idx) + 2}: Invalid service '{service_val}' for {carrier_val}. Valid: {valid_services}")
                return None
        else:
            # Auto-detect carrier from service code
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
        st.error("❌ No valid service codes found. Use codes like: F2, P, U, or leave empty for RS")
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
        
        # Handle order_id
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
            st.error(f"❌ Row {int(idx) + 2}: Missing required field")  # ✅ FIXED: int(idx)
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

        # Get valid lwh values
        lwh_values = {}
        for lwh_col in lwh_columns:
            val = safe_float(lwh_col)
            if val is not None and val > 0:
                lwh_values[lwh_col] = val

        # Get valid weight values
        weight_values = {}
        for weight_col in weight_columns:
            val = safe_float(weight_col)
            if val is not None and val > 0:
                weight_values[weight_col] = val
        
        if not weight_values:
            weight_values = {"weight": 1.0}
            st.warning(f"⚠️ Row {int(idx) + 2}: No valid weights. Using default = 1.0")  # ✅ FIXED: int(idx)

        # Create package for every lwh-weight combination
        for lwh_col, lwh_val in lwh_values.items():
            for weight_col, weight_val in weight_values.items():
                for box_num in range(num_boxes):
                    unique_shipment_id = str(uuid.uuid4()).replace("-", "")[:16]
                    packages.append({
                        "SKU": safe_string('sku', "N/A"),
                        "Weight": weight_val,
                        "Description": safe_string('description', "No description"),
                        "Length": lwh_val,
                        "Width": lwh_val,
                        "Height": lwh_val,
                        "PackagingWeight": packaging_weight,
                        "Address": address,
                        "ServiceLevel": service_level,
                        "Carrier": carrier,
                        "ClientCode": row_client_code,
                        "LWH_Source": lwh_col,
                        "Weight_Source": weight_col,
                        "Box_Index": box_num,
                        "UNIQUE_SHIPMENT_ID": unique_shipment_id,
                        "OrderID": user_order_id
                    })

    if not packages:
        st.error("❌ No valid packages created.")
        return None
    return packages, carrier

def submit_single_shipment(payload, client_code):
    session = create_robust_session()
    original_payload = payload.copy()
    try:
        payload["ClientCode"] = client_code
        response = session.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        
        if response.status_code == 409:
            return {
                "Status": "❌ Duplicate (409)",
                "TransactionNumber": payload["TransactionNumber"],
                "TrackingNumber": "",
                "Cost": "$0.00",
                "Service": payload["Routing"]["ServiceCode"],
                "Recipient": payload["ShipToAddress"]["Name"],
                "PostalCode": payload["ShipToAddress"]["Postal"],
                "Boxes": len(payload["Packages"]),
                "Error": "409 Conflict: Duplicate shipment",
                "_original_payload": original_payload,
                "ClientCode": client_code
            }
        elif response.status_code == 500:
            error_text = response.text[:100] if response.text else "No details"
            return {
                "Status": "❌ Server Error (500)",
                "TransactionNumber": payload["TransactionNumber"],
                "TrackingNumber": "",
                "Cost": "$0.00",
                "Service": payload["Routing"]["ServiceCode"],
                "Recipient": payload["ShipToAddress"]["Name"],
                "PostalCode": payload["ShipToAddress"]["Postal"],
                "Boxes": len(payload["Packages"]),
                "Error": f"500: {error_text}",
                "_original_payload": original_payload,
                "ClientCode": client_code
            }
        
        response.raise_for_status()
        try:
            response_data = response.json()
        except ValueError:
            return {
                "Status": "❌ Invalid Response",
                "TransactionNumber": payload["TransactionNumber"],
                "TrackingNumber": "",
                "Cost": "$0.00",
                "Service": payload["Routing"]["ServiceCode"],
                "Recipient": payload["ShipToAddress"]["Name"],
                "PostalCode": payload["ShipToAddress"]["Postal"],
                "Boxes": len(payload["Packages"]),
                "Error": "Non-JSON response",
                "_original_payload": original_payload,
                "ClientCode": client_code
            }
        
        label_info = response_data.get("Labels", [{}])[0]
        total_cost = response_data.get("ShipmentTotalCharge", 0)
        return {
            "Status": "✅ Success",
            "TransactionNumber": payload["TransactionNumber"],
            "TrackingNumber": label_info.get("TrackingNumber", "N/A"),
            "Cost": f"${total_cost:.2f}",
            "Service": payload["Routing"]["ServiceCode"],
            "Recipient": payload["ShipToAddress"]["Name"],
            "PostalCode": payload["ShipToAddress"]["Postal"],
            "Boxes": len(payload["Packages"]),
            "LabelURL": label_info.get("LabelUrl", ""),
            "_original_payload": original_payload,
            "ClientCode": client_code
        }
    except Exception as e:
        error_msg = str(e)
        return {
            "Status": "❌ Failed",
            "TransactionNumber": payload.get("TransactionNumber", "unknown"),
            "TrackingNumber": "",
            "Cost": "$0.00",
            "Service": payload.get("Routing", {}).get("ServiceCode", "unknown"),
            "Recipient": payload.get("ShipToAddress", {}).get("Name", "unknown"),
            "PostalCode": payload.get("ShipToAddress", {}).get("Postal", "unknown"),
            "Boxes": len(payload.get("Packages", [])),
            "Error": error_msg[:100],
            "_original_payload": original_payload,
            "ClientCode": client_code
        }
    finally:
        session.close()

def submit_concurrent_shipments(packages, carrier, fallback_client_code, max_workers=3):
    actual_workers = min(max_workers, 4)
    shipment_groups = defaultdict(list)
    for package in packages:
        # Include weight in grouping key
        key = (
            package["Address"]["Name"],
            package["Address"]["Address1"], 
            package["Address"]["City"],
            package["Address"]["Postal"],
            package["ServiceLevel"],
            package["Length"],
            package["Width"], 
            package["Height"],
            package["Weight"],          # ✅ Critical for weight variants
            package["PackagingWeight"]
        )
        shipment_groups[key].append(package)
    
    batch_id = str(uuid.uuid4()).replace("-", "")[:20]
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
        payload["_client_code"] = client_code_val
        
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
        payloads.append(payload)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        for payload in payloads:
            client_code_for_shipment = payload.get("_client_code", fallback_client_code)
            future = executor.submit(submit_single_shipment, payload, client_code_for_shipment)
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
    st.set_page_config(page_title="TechSHIP Bulk Shipper", page_icon="📦", layout="wide")
    st.title("📦 TechSHIP Bulk Shipment Processor")
    st.markdown("### Compare pricing across box sizes and weights")

    fallback_client_code = st.text_input("Fallback Client Code", value="omrtest1", 
                                        help="Used if 'client_code' column is missing in input data")
    if not fallback_client_code.strip():
        st.warning("⚠️ Please enter a valid Fallback Client Code")
        st.stop()

    # ✅ NEW: Add "Trigger RS" button
    col1, col2 = st.columns(2)
    with col1:
        trigger_rs = st.button("🎯 Trigger RS (RateShopping)", type="secondary")
    with col2:
        pass  # Empty column for spacing

    with st.sidebar:
        st.header("📋 Usage Guide")
        st.markdown(f"""
        **Required Columns**: `name`, `services`
        **Optional Columns**:
        - `carrier`: Use `RS` for RateShopping, `FEDEX`, `PURO`, or `UPS`
        - `services`: Leave **blank** for RS, or use codes like `F2`, `P`, `U`
        - `lwh`, `lwh2`, ...: Box sizes
        - `weight`, `weight2`, ...: Weights
        - `postal_prefix`: Auto-fills address (e.g., B2J)
        """)
        st.info("✅ RS is triggered by blank `services` OR `carrier = 'RS'`")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
    with col2:
        text_input = st.text_area("📋 Or Paste Data", height=150)

    if "all_results" not in st.session_state:
        st.session_state.all_results = []
        st.session_state.batch_id = ""

    if st.button("🚀 Process & Submit to TechSHIP", type="primary"):
        with st.spinner("🔍 Parsing data..."):
            df = parse_input_data(text_input, uploaded_file)
            if df is None: st.stop()
            result = validate_and_process_data(df, fallback_client_code.strip(), force_rs=trigger_rs)
            if result is None: st.stop()
            packages, carrier = result
            st.success(f"✅ Parsed {len(packages)} packages for {carrier}")

        st.subheader("⚙️ Configuration")
        max_workers = st.slider("ParallelGroups", 1, 4, 3, help="Keep at 3 to avoid 500 errors")
        st.subheader("📤 Submitting...")
        try:
            results, batch_id = submit_concurrent_shipments(packages, carrier, fallback_client_code.strip(), max_workers)
            st.session_state.all_results = results
            st.session_state.batch_id = batch_id
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    if st.session_state.all_results:
        results_list = st.session_state.all_results
        success_count = sum(1 for r in results_list if "Success" in r.get("Status", ""))
        failed_count = len(results_list) - success_count

        st.subheader("📊 Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results_list))
        col2.metric("Success", success_count)
        col3.metric("Failed", failed_count)

        display_data = []
        for r in results_list:
            row = {
                "Status": r.get("Status", "Unknown"),
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
        display_cols = ["Status", "Boxes", "TrackingNumber", "Cost", "Service", "Recipient", "PostalCode"]
        if "Error" in results_df.columns:
            display_cols.append("Error")
        st.dataframe(results_df[display_cols], use_container_width=True)

        has_failed = any("409" in r.get("Status", "") or "500" in r.get("Status", "") or "Failed" in r.get("Status", "") for r in results_list)
        if has_failed:
            if st.button("🔁 Retry Failed Shipments", type="secondary"):
                new_results = []
                for result in results_list:
                    status = result.get("Status", "")
                    if "409" in status or "500" in status or "Failed" in status:
                        if "_original_payload" in result:
                            payload = result["_original_payload"].copy()
                            payload["TransactionNumber"] = str(uuid.uuid4()).replace("-", "")[:20]
                            order_id = None
                            for pkg in payload["Packages"]:
                                if "OrderID" in pkg:
                                    order_id = pkg["OrderID"]
                                    break
                            if not order_id:
                                order_id = str(uuid.uuid4()).replace("-", "")[:20]
                            payload["CustomerOrder"] = order_id[:20]
                            
                            client_code_retry = result.get("ClientCode", fallback_client_code.strip())
                            new_result = submit_single_shipment(payload, client_code_retry)
                            new_results.append(new_result)
                        else:
                            new_results.append(result)
                    else:
                        new_results.append(result)
                st.session_state.all_results = new_results
                st.rerun()

        csv = pd.DataFrame(display_data).to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Results", csv, f"techship_{st.session_state.batch_id}.csv", "text/csv")

if __name__ == "__main__":
    main()