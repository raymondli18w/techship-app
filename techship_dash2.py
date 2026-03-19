import streamlit as st
import pandas as pd
import requests
import uuid
import concurrent.futures
import time
import math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO, StringIO
from collections import defaultdict

# SQLite Address Lookup
from sqlite_lookup import get_address_by_prefix

# =========================
# MUST be the first Streamlit command
# =========================
st.set_page_config(page_title="TechSHIP Bulk Rate Estimator", page_icon="📦", layout="wide")

# =========================
# TechSHIP API Configuration
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
def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

def safe_string(value, default=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return str(value).strip()

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=200, pool_maxsize=200)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def submit_chunk(payload, client_code, order_id, batch_id, dry_run=True, chunk_num=1, total_chunks=1, max_retries=3):
    """Submit chunk with retry logic for HTTP 500 errors"""
    session = create_robust_session()
    timeout = 60
    
    try:
        for attempt in range(max_retries):
            try:
                payload["ClientCode"] = client_code
                params = {"dryRun": "true" if dry_run else "false"}
                
                response = session.post(API_URL, headers=HEADERS, json=payload, params=params, timeout=timeout)

                # SUCCESS
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if not isinstance(response_data, dict):
                            response_data = {}
                    except Exception:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {
                            "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                            "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                            "error": "Invalid JSON response", "boxes": len(payload.get("Packages", [])),
                            "chunk_num": chunk_num, "total_chunks": total_chunks
                        }

                    rates = response_data.get("Rates", [])
                    if rates and len(rates) > 0:
                        best_rate = next((r for r in rates if r.get("IsBest")), rates[0])
                        return {
                            "success": True,
                            "cost": safe_float(best_rate.get("TotalAmount", best_rate.get("Amount", 0))),
                            "base_amount": safe_float(best_rate.get("BaseAmount", 0)),
                            "fuel_surcharge": safe_float(best_rate.get("FuelSurcharge", 0)),
                            "public_total": safe_float(best_rate.get("PublicTotalAmount", 0)),
                            "service": best_rate.get("ServiceName", best_rate.get("ServiceCode", "N/A")),
                            "carrier": payload.get("CarrierCode", "N/A"), "error": None,
                            "boxes": len(payload.get("Packages", [])), "chunk_num": chunk_num, "total_chunks": total_chunks
                        }
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return {
                            "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                            "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                            "error": "No rates returned", "boxes": len(payload.get("Packages", [])),
                            "chunk_num": chunk_num, "total_chunks": total_chunks
                        }

                # HTTP 500 - RETRY
                elif response.status_code == 500:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    else:
                        error_text = response.text[:300] if response.text else "Server error"
                        return {
                            "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                            "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                            "error": f"HTTP 500: {error_text} (after {max_retries} retries)",
                            "boxes": len(payload.get("Packages", [])),
                            "chunk_num": chunk_num, "total_chunks": total_chunks
                        }

                # OTHER ERRORS
                else:
                    error_text = response.text[:300] if response.text else "No details"
                    return {
                        "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                        "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                        "error": f"HTTP {response.status_code}: {error_text}",
                        "boxes": len(payload.get("Packages", [])),
                        "chunk_num": chunk_num, "total_chunks": total_chunks
                    }

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                    "public_total": 0.0, "service": "N/A", "carrier": "N/A", "error": f"Timeout after {timeout}s",
                    "boxes": 0, "chunk_num": chunk_num, "total_chunks": total_chunks
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                    "public_total": 0.0, "service": "N/A", "carrier": "N/A", "error": str(e)[:200],
                    "boxes": 0, "chunk_num": chunk_num, "total_chunks": total_chunks
                }
        
        # All retries exhausted
        return {
            "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
            "public_total": 0.0, "service": "N/A", "carrier": "N/A", "error": "All retries exhausted",
            "boxes": 0, "chunk_num": chunk_num, "total_chunks": total_chunks
        }
    finally:
        session.close()

def process_single_order(row, fallback_client_code, batch_id, dry_run=True, chunk_size=10):
    """Process a single order with automatic splitting for large box counts"""
    try:
        num_boxes = safe_float(row.get('boxes', 1), 1.0)
        if num_boxes < 1:
            num_boxes = 1
        if num_boxes > 100:
            num_boxes = 100
        
        num_chunks = math.ceil(num_boxes / chunk_size)
        
        weight = safe_float(row.get('weight', 1), 1.0)
        length = safe_float(row.get('length') or row.get('lwh', 10), 10)
        width = safe_float(row.get('width') or row.get('lwh', 10), 10)
        height = safe_float(row.get('height') or row.get('lwh', 10), 10)
        
        if length > 1000 or width > 1000 or height > 1000:
            length = safe_float(row.get('length', 10), 10)
            width = safe_float(row.get('width', 10), 10)
            height = safe_float(row.get('height', 10), 10)
        
        city = safe_string(row.get('city', ''), 'Toronto')
        if not city:
            city = "Toronto"
        
        province = safe_string(row.get('province', 'ON'), 'ON').upper()[:2]
        if province not in ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']:
            province = 'ON'
        
        country = safe_string(row.get('country', 'CA'), 'CA').upper()
        if not country or country not in ['CA', 'CAN', 'USA', 'US']:
            country = "CA"
        
        postal = safe_string(row.get('postal', ''), 'M5V1J1').replace(" ", "").upper()[:10]
        if len(postal) < 5:
            postal = "M5V1J1"
        
        email = safe_string(row.get('email', ''), 'test@test.com')
        if not email or '@' not in email:
            email = "test@test.com"
        
        carrier_input = safe_string(row.get('carrier', ''), '').upper()
        service_level = safe_string(row.get('services', ''), '')
        
        if carrier_input == 'RS' or (carrier_input == '' and service_level == ''):
            actual_carrier_code = "RS"
            service_code = ""
        elif carrier_input in CARRIER_SERVICE_MAP:
            actual_carrier_code = CARRIER_SERVICE_MAP[carrier_input]["CarrierCode"]
            service_code = service_level if service_level else ""
        elif service_level in SERVICE_TO_CARRIER:
            actual_carrier_code = CARRIER_SERVICE_MAP[SERVICE_TO_CARRIER[service_level]]["CarrierCode"]
        else:
            actual_carrier_code = "RS"
            service_code = ""
        
        chunk_results = []
        transaction_number = str(uuid.uuid4()).replace("-", "")[:20]
        customer_order = safe_string(row.get('order_id', f'ORD-{uuid.uuid4().hex[:8]}'))[:20]
        
        for chunk_num in range(1, num_chunks + 1):
            boxes_in_chunk = int(min(chunk_size, num_boxes - ((chunk_num - 1) * chunk_size)))
            
            packages_array = []
            for box in range(boxes_in_chunk):
                packages_array.append({
                    "Weight": weight,
                    "Dimensions": {
                        "Length": length,
                        "Width": width,
                        "Height": height,
                        "PackagingWeight": safe_float(row.get('packaging_weight', 0), 0.0)
                    },
                    "Items": [{
                        "SKU": safe_string(row.get('sku', 'N/A')),
                        "Description": safe_string(row.get('description', 'No description')),
                        "Quantity": 1
                    }]
                })
            
            payload = {
                "TransactionNumber": f"{transaction_number}-{chunk_num:03d}",
                "CustomerOrder": customer_order,
                "BatchNumber": batch_id,
                "CarrierCode": actual_carrier_code,
                "Routing": {
                    "CarrierCode": actual_carrier_code,
                    "ServiceCode": service_code,
                    "FreightPaymentTerms": "Prepaid"
                },
                "ShipToAddress": {
                    "Name": safe_string(row.get('name', 'John Doe')),
                    "Company": safe_string(row.get('company', '')),
                    "Address1": safe_string(row.get('address', '')),
                    "Address2": safe_string(row.get('address2', '')),
                    "City": city,
                    "StateProvince": province,
                    "Postal": postal,
                    "Country": country,
                    "Phone": safe_string(row.get('phone', '6045555555')),
                    "Email": email
                },
                "Packages": packages_array
            }
            
            client_code_val = safe_string(row.get('client_code', fallback_client_code)) or fallback_client_code
            
            result = submit_chunk(payload, client_code_val, customer_order, batch_id, dry_run, chunk_num, num_chunks, max_retries=3)
            chunk_results.append(result)
        
        total_cost = sum(safe_float(r.get("cost", 0)) for r in chunk_results if r.get("success"))
        total_base = sum(safe_float(r.get("base_amount", 0)) for r in chunk_results if r.get("success"))
        total_fuel = sum(safe_float(r.get("fuel_surcharge", 0)) for r in chunk_results if r.get("success"))
        total_public = sum(safe_float(r.get("public_total", 0)) for r in chunk_results if r.get("success"))
        successful_chunks = sum(1 for r in chunk_results if r.get("success"))
        
        service_info = next((r.get("service", "N/A") for r in chunk_results if r.get("success")), "N/A")
        carrier_info = next((r.get("carrier", actual_carrier_code) for r in chunk_results if r.get("success")), actual_carrier_code)
        
        error_info = None
        failed_chunks = num_chunks - successful_chunks
        if failed_chunks > 0:
            errors = [f"Chunk {r.get('chunk_num', '?')}: {r.get('error', 'Unknown')}" for r in chunk_results if not r.get("success") and r.get("error")]
            error_info = "; ".join(errors[:2])
        
        chunk_display = f"{successful_chunks}/{num_chunks}" if num_chunks > 1 else "1/1"
        
        return {
            "Status": "✅ Estimate" if successful_chunks == num_chunks else f"⚠️ Partial ({successful_chunks}/{num_chunks})",
            "OrderID": customer_order,
            "TransactionNumber": transaction_number,
            "BatchID": batch_id,
            "Boxes": int(num_boxes),
            "Chunks": chunk_display,
            "Cost": f"${total_cost:.2f}",
            "BaseAmount": f"${total_base:.2f}",
            "FuelSurcharge": f"${total_fuel:.2f}",
            "PublicTotal": f"${total_public:.2f}",
            "Service": service_info,
            "Carrier": carrier_info,
            "Recipient": safe_string(row.get('name', 'John Doe')),
            "City": city,
            "Province": province,
            "PostalCode": postal,
            "Email": email,
            "Error": error_info,
            "DryRun": dry_run
        }
    except Exception as e:
        return {
            "Status": "❌ Error",
            "OrderID": safe_string(row.get('order_id', f'ORD-?')),
            "TransactionNumber": "N/A",
            "BatchID": batch_id,
            "Boxes": 0,
            "Chunks": "0/0",
            "Cost": "$0.00",
            "BaseAmount": "$0.00",
            "FuelSurcharge": "$0.00",
            "PublicTotal": "$0.00",
            "Service": "N/A",
            "Carrier": "N/A",
            "Recipient": "N/A",
            "City": "N/A",
            "Province": "N/A",
            "PostalCode": "N/A",
            "Email": "N/A",
            "Error": str(e)[:200],
            "DryRun": dry_run
        }

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
    st.markdown("### Rate Limit Protected — Auto-Retry on HTTP 500")

    fallback_client_code = st.text_input("Fallback Client Code", value="8470HWY50")
    if not fallback_client_code.strip():
        st.warning("⚠️ Please enter a valid Fallback Client Code")
        st.stop()

    dry_run = st.checkbox("🔒 Dry Run Mode (Estimates Only)", value=True)
    chunk_size = st.sidebar.slider("📦 Boxes Per API Call", 5, 50, 10)

    with st.sidebar:
        st.header("📊 Progress")
        st.info("""
        **Rate Limit Protection:**
        1. Auto-retry on HTTP 500 (3 attempts)
        2. Exponential backoff (2s, 4s, 8s)
        3. Single worker to prevent overload
        4. Longer delay between orders
        """)
        
        auto_continue = st.checkbox("🔄 Auto-Continue Mode", value=False)
        
        if auto_continue:
            st.success("🔄 **Auto-Continue ON**")
        else:
            st.warning("⏸️ **Auto-Continue OFF**")
        
        delay_seconds = st.slider("⏱️ Delay Between Orders (seconds)", 3, 30, 10,
                                  help="Higher = more stable for 40,000 orders")
        
        if dry_run:
            st.warning("⚠️ Dry Run ON")
        else:
            st.success("✅ Will Save to DB")
        
        st.markdown("---")
        st.code("https://18wheels.techship.ca/")

    # INITIALIZE SESSION STATE
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "all_orders" not in st.session_state:
        st.session_state.all_orders = []
    if "processed_indices" not in st.session_state:
        st.session_state.processed_indices = []
    if "all_results" not in st.session_state:
        st.session_state.all_results = []
    if "batch_id" not in st.session_state:
        st.session_state.batch_id = ""
    if "total_orders" not in st.session_state:
        st.session_state.total_orders = 0
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "last_process_time" not in st.session_state:
        st.session_state.last_process_time = 0
    if "consecutive_500_errors" not in st.session_state:
        st.session_state.consecutive_500_errors = 0

    # FILE UPLOAD
    if not st.session_state.file_uploaded:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
        with col2:
            text_input = st.text_area("📋 Or Paste Data", height=150)
        
        if st.button("🚀 Load File", type="primary"):
            with st.spinner("🔍 Parsing file..."):
                try:
                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
                        else:
                            df = pd.read_excel(BytesIO(uploaded_file.getvalue()))
                    elif text_input.strip():
                        delimiter = '\t' if '\t' in text_input[:500] else ','
                        df = pd.read_csv(StringIO(text_input), delimiter=delimiter)
                    else:
                        st.error("❌ Please upload file or paste data")
                        st.stop()
                    
                    df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
                    df = df.loc[:, df.columns != '']
                    
                    if 'name' not in df.columns or 'services' not in df.columns:
                        st.error("❌ Missing required columns: name, services")
                        st.stop()
                    
                    st.session_state.all_orders = df.to_dict('records')
                    st.session_state.total_orders = len(df)
                    st.session_state.file_uploaded = True
                    st.session_state.batch_id = str(uuid.uuid4()).replace("-", "")[:20]
                    
                    st.success(f"✅ Loaded {st.session_state.total_orders} orders!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Parse error: {str(e)}")
    else:
        processed = len(st.session_state.processed_indices)
        total = st.session_state.total_orders
        remaining = total - processed
        percent = (processed / total * 100) if total > 0 else 0
        
        st.success(f"✅ File loaded: {total} orders")
        st.progress(percent / 100)
        st.metric("Progress", f"{processed}/{total} orders ({percent:.1f}%)")
        
        if st.session_state.all_results:
            last = st.session_state.all_results[-1]
            st.info(f"📦 Last: {last.get('OrderID', 'N/A')} - {last.get('City')}, {last.get('Province')} - {last.get('Cost')} - {last.get('Status')}")
        
        # RATE LIMIT WARNING
        if st.session_state.consecutive_500_errors >= 5:
            st.error(f"⚠️ **{st.session_state.consecutive_500_errors} consecutive HTTP 500 errors.** TechSHIP API may be overloaded. Consider:")
            st.write("1. Increase delay between orders (sidebar)")
            st.write("2. Pause and resume later")
            st.write("3. Contact TechSHIP support about rate limits")
        
        # AUTO-CONTINUE LOGIC
        if auto_continue and remaining > 0 and not st.session_state.processing_complete:
            current_time = time.time()
            time_since_last = current_time - st.session_state.last_process_time
            
            if time_since_last >= delay_seconds or st.session_state.last_process_time == 0:
                idx = processed
                row = st.session_state.all_orders[idx]
                
                status_container = st.empty()
                with status_container:
                    st.info(f"⏳ Processing order {idx + 1} of {total}...")
                
                result = process_single_order(row, fallback_client_code.strip(), st.session_state.batch_id, dry_run, chunk_size)
                
                st.session_state.all_results.append(result)
                st.session_state.processed_indices.append(idx)
                st.session_state.last_process_time = time.time()
                
                # Track consecutive 500 errors
                if "HTTP 500" in result.get("Error", ""):
                    st.session_state.consecutive_500_errors += 1
                else:
                    st.session_state.consecutive_500_errors = 0
                
                with status_container:
                    if result.get("Status", "").startswith("✅"):
                        st.success(f"✅ {result.get('OrderID')}: {result.get('City')}, {result.get('Province')} - {result.get('Cost')} ({idx + 1}/{total})")
                    else:
                        st.warning(f"⚠️ {result.get('OrderID')}: {result.get('Error', 'Failed')[:150]}")
                
                if len(st.session_state.processed_indices) >= total:
                    st.session_state.processing_complete = True
                    st.balloons()
                    st.success(f"🎉 All {total} orders processed!")
                
                if not st.session_state.processing_complete:
                    time.sleep(0.5)
                    st.rerun()
            else:
                countdown = int(delay_seconds - time_since_last)
                st.info(f"⏱️ Next order in {countdown} seconds... ({processed + 1}/{total})")
                time.sleep(1)
                st.rerun()
        
        # MANUAL BUTTONS
        if not auto_continue or st.session_state.processing_complete:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if remaining > 0 and not st.session_state.processing_complete:
                    if st.button("▶️ Process 1 Order", type="primary" if not auto_continue else "secondary", use_container_width=True):
                        idx = processed
                        row = st.session_state.all_orders[idx]
                        
                        with st.spinner(f"⏳ Processing order {idx + 1} of {total}..."):
                            result = process_single_order(row, fallback_client_code.strip(), st.session_state.batch_id, dry_run, chunk_size)
                            
                            st.session_state.all_results.append(result)
                            st.session_state.processed_indices.append(idx)
                            
                            if "HTTP 500" in result.get("Error", ""):
                                st.session_state.consecutive_500_errors += 1
                            else:
                                st.session_state.consecutive_500_errors = 0
                            
                            if result.get("Status", "").startswith("✅"):
                                st.success(f"✅ {result.get('OrderID')}: {result.get('Cost')}")
                            else:
                                st.error(f"❌ {result.get('OrderID')}: {result.get('Error', 'Failed')[:200]}")
                            
                            if len(st.session_state.processed_indices) >= total:
                                st.session_state.processing_complete = True
                                st.balloons()
                            
                            st.rerun()
            
            with col2:
                if remaining > 0 and not st.session_state.processing_complete:
                    if st.button("▶️ Process 10 Orders", use_container_width=True):
                        start_idx = processed
                        end_idx = min(start_idx + 10, total)
                        
                        with st.spinner(f"⏳ Processing orders {start_idx + 1}-{end_idx}..."):
                            for idx in range(start_idx, end_idx):
                                row = st.session_state.all_orders[idx]
                                result = process_single_order(row, fallback_client_code.strip(), st.session_state.batch_id, dry_run, chunk_size)
                                st.session_state.all_results.append(result)
                                st.session_state.processed_indices.append(idx)
                                
                                if "HTTP 500" in result.get("Error", ""):
                                    st.session_state.consecutive_500_errors += 1
                                else:
                                    st.session_state.consecutive_500_errors = 0
                            
                            if len(st.session_state.processed_indices) >= total:
                                st.session_state.processing_complete = True
                                st.balloons()
                            
                            st.rerun()
            
            with col3:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            with col4:
                if st.session_state.all_results:
                    csv = pd.DataFrame(st.session_state.all_results).to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download", csv, f"techship_{st.session_state.batch_id}.csv", "text/csv", use_container_width=True)
        
        # AUTO-CONTINUE CONTROLS
        if auto_continue and remaining > 0 and not st.session_state.processing_complete:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ Pause Auto-Continue", type="secondary", use_container_width=True):
                    auto_continue = False
                    st.rerun()
            with col2:
                if st.button("💾 Download Progress Now", use_container_width=True):
                    csv = pd.DataFrame(st.session_state.all_results).to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Partial Results", csv, f"techship_partial_{st.session_state.batch_id}.csv", "text/csv", use_container_width=True)
        
        # RESET BUTTON
        if st.button("🗑️ Reset & New File"):
            for key in ["file_uploaded", "all_orders", "processed_indices", "all_results", "batch_id", "total_orders", "processing_complete", "last_process_time", "consecutive_500_errors"]:
                st.session_state[key] = [] if key in ["all_orders", "processed_indices", "all_results"] else "" if key == "batch_id" else False if key in ["file_uploaded", "processing_complete"] else 0 if key == "total_orders" else 0
            st.rerun()
        
        # DISPLAY RESULTS
        if st.session_state.all_results:
            st.subheader(f"📊 Results ({len(st.session_state.all_results)} orders)")
            
            results_df = pd.DataFrame(st.session_state.all_results)
            display_cols = ["OrderID", "Status", "City", "Province", "Boxes", "Chunks", "Cost", "Service", "Carrier"]
            if "Error" in results_df.columns:
                display_cols.append("Error")
            
            st.dataframe(results_df[display_cols].tail(50), use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            success = sum(1 for r in st.session_state.all_results if "✅" in r.get("Status", ""))
            col1.metric("Successful", success)
            col2.metric("Failed", len(st.session_state.all_results) - success)
            total_cost = sum(safe_float(r.get('Cost', '$0').replace('$', '')) for r in st.session_state.all_results)
            col3.metric("Total Cost", f"${total_cost:.2f}")
        
        # REMAINING INFO
        if remaining > 0:
            if auto_continue:
                st.info(f"🔄 **Auto-Processing:** {remaining} orders remaining...")
                est_time = remaining * delay_seconds
                st.write(f"⏱️ Est. time: ~{est_time // 60} min {est_time % 60} sec")
            else:
                st.info(f"⏭️ {remaining} orders remaining.")
        else:
            st.success("🎉 All orders processed!")
            
            if st.session_state.all_results:
                csv = pd.DataFrame(st.session_state.all_results).to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Final Report", csv, f"techship_FINAL_{st.session_state.batch_id}.csv", "text/csv", type="primary", use_container_width=True)

if __name__ == "__main__":
    main()
