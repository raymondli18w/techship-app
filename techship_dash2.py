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
    """Safely convert value to float, handling NaN/None/invalid values"""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=200, pool_maxsize=200)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def submit_chunk(payload, client_code, order_id, batch_id, dry_run=True, chunk_num=1, total_chunks=1):
    session = create_robust_session()
    timeout = 120
    
    try:
        payload["ClientCode"] = client_code
        params = {"dryRun": "true" if dry_run else "false"}
        response = session.post(API_URL, headers=HEADERS, json=payload, params=params, timeout=timeout)

        if response.status_code != 200:
            error_text = response.text[:300] if response.text else "No details"
            return {
                "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                "error": f"HTTP {response.status_code}: {error_text}", "boxes": len(payload.get("Packages", [])),
                "chunk_num": chunk_num, "total_chunks": total_chunks
            }

        try:
            response_data = response.json()
            if not isinstance(response_data, dict):
                response_data = {}
        except Exception:
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
            # ✅ Better error message showing WHY no rates
            return {
                "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
                "public_total": 0.0, "service": "N/A", "carrier": payload.get("CarrierCode", "N/A"),
                "error": f"No rates returned. Check: Address valid? Service code correct? Carrier active?",
                "boxes": len(payload.get("Packages", [])),
                "chunk_num": chunk_num, "total_chunks": total_chunks
            }
    except Exception as e:
        return {
            "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0,
            "public_total": 0.0, "service": "N/A", "carrier": "N/A", "error": str(e)[:150],
            "boxes": 0, "chunk_num": chunk_num, "total_chunks": total_chunks
        }
    finally:
        session.close()

def process_order_batch(order_packages, carrier, fallback_client_code, batch_id, dry_run=True, chunk_size=50):
    """Process a single order with chunking for large box counts"""
    total_boxes = len(order_packages)
    num_chunks = max(1, (total_boxes + chunk_size - 1) // chunk_size)
    
    chunk_results = []
    transaction_number = str(uuid.uuid4()).replace("-", "")[:20]
    customer_order = order_packages[0]["OrderID"][:20]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, num_chunks)) as executor:
        futures = []
        for chunk_num, start_idx in enumerate(range(0, total_boxes, chunk_size), 1):
            chunk_packages = order_packages[start_idx:start_idx + chunk_size]
            packages_array = [{
                "Weight": pkg["Weight"],
                "Dimensions": {"Length": pkg["Length"], "Width": pkg["Width"], "Height": pkg["Height"], "PackagingWeight": pkg["PackagingWeight"]},
                "Items": [{"SKU": pkg["SKU"], "Description": pkg["Description"], "Quantity": 1}]
            } for pkg in chunk_packages]
            
            payload = {
                "TransactionNumber": f"{transaction_number}-{chunk_num:03d}",
                "CustomerOrder": customer_order,
                "BatchNumber": batch_id,
                "CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"],
                "Routing": {"CarrierCode": CARRIER_SERVICE_MAP[carrier]["CarrierCode"], "ServiceCode": chunk_packages[0]["ServiceLevel"], "FreightPaymentTerms": "Prepaid"},
                "ShipToAddress": chunk_packages[0]["Address"],
                "Packages": packages_array
            }
            
            client_code_val = chunk_packages[0].get("ClientCode") or fallback_client_code
            future = executor.submit(submit_chunk, payload, client_code_val, customer_order, batch_id, dry_run, chunk_num, num_chunks)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            chunk_results.append(result if result and isinstance(result, dict) else {
                "success": False, "cost": 0.0, "base_amount": 0.0, "fuel_surcharge": 0.0, "public_total": 0.0,
                "service": "N/A", "carrier": carrier, "error": "No result", "boxes": 0,
                "chunk_num": len(chunk_results) + 1, "total_chunks": num_chunks
            })
    
    total_cost = sum(safe_float(r.get("cost", 0)) for r in chunk_results if r.get("success"))
    total_base = sum(safe_float(r.get("base_amount", 0)) for r in chunk_results if r.get("success"))
    total_fuel = sum(safe_float(r.get("fuel_surcharge", 0)) for r in chunk_results if r.get("success"))
    total_public = sum(safe_float(r.get("public_total", 0)) for r in chunk_results if r.get("success"))
    successful_chunks = sum(1 for r in chunk_results if r.get("success"))
    
    service_info = next((r.get("service", "N/A") for r in chunk_results if r.get("success")), "N/A")
    carrier_info = next((r.get("carrier", carrier) for r in chunk_results if r.get("success")), carrier)
    
    error_info = None
    failed_chunks = num_chunks - successful_chunks
    if failed_chunks > 0:
        errors = [f"Chunk {r.get('chunk_num', '?')}: {r.get('error', 'Unknown')}" for r in chunk_results if not r.get("success") and r.get("error")]
        error_info = "; ".join(errors[:3])
    
    return {
        "Status": "✅ Estimate" if successful_chunks == num_chunks else f"⚠️ Partial ({successful_chunks}/{num_chunks})",
        "OrderID": customer_order, "TransactionNumber": transaction_number, "BatchID": batch_id,
        "Boxes": total_boxes, "Cost": f"${total_cost:.2f}", "BaseAmount": f"${total_base:.2f}",
        "FuelSurcharge": f"${total_fuel:.2f}", "PublicTotal": f"${total_public:.2f}",
        "Service": service_info, "Carrier": carrier_info,
        "Recipient": order_packages[0]["Address"]["Name"], "PostalCode": order_packages[0]["Address"]["Postal"],
        "Chunks": f"{successful_chunks}/{num_chunks}", "Error": error_info, "DryRun": dry_run
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
    st.markdown("### ⚡ Incremental Batch Processing — See Results Immediately")

    fallback_client_code = st.text_input("Fallback Client Code", value="8470HWY50")
    if not fallback_client_code.strip():
        st.warning("⚠️ Please enter a valid Fallback Client Code")
        st.stop()

    dry_run = st.checkbox("🔒 Dry Run Mode (Estimates Only)", value=True)
    batch_size = st.sidebar.slider("📦 Orders Per Batch", 50, 500, 100)
    chunk_size = st.sidebar.slider("📦 Boxes Per API Call", 25, 100, 50)

    with st.sidebar:
        st.header("📊 Progress")
        st.info("""
        **How It Works:**
        1. Upload file ONCE
        2. Process 100 orders at a time
        3. See results immediately
        4. Click Continue for next batch
        5. Download anytime
        """)
        
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
    if "carrier" not in st.session_state:
        st.session_state.carrier = ""
    if "total_orders" not in st.session_state:
        st.session_state.total_orders = 0
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    # FILE UPLOAD (Only Once)
    if not st.session_state.file_uploaded:
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
        with col2:
            text_input = st.text_area("📋 Or Paste Data", height=150)
        
        if st.button("🚀 Load & Start Processing", type="primary"):
            with st.spinner("🔍 Parsing file (one time only)..."):
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
                    
                    # Clean columns
                    df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
                    df = df.loc[:, df.columns != '']
                    
                    # Validate
                    if 'name' not in df.columns or 'services' not in df.columns:
                        st.error("❌ Missing required columns: name, services")
                        st.stop()
                    
                    # ✅ DATA QUALITY CHECK - Show warnings before processing
                    issues = []
                    if 'city' in df.columns:
                        empty_cities = df['city'].isna().sum() if 'city' in df.columns else len(df)
                        if empty_cities > len(df) * 0.5:
                            issues.append(f"⚠️ {empty_cities}/{len(df)} rows have empty city - may cause rate failures")
                    
                    if 'lwh' in df.columns:
                        lwh_max = df['lwh'].max() if pd.notna(df['lwh']).any() else 0
                        if lwh_max > 1000:
                            issues.append(f"⚠️ lwh column has unrealistic values (max: {lwh_max}) - will use length/width/height instead")
                    
                    if issues:
                        st.warning("📋 **Data Quality Issues Found:**\n\n" + "\n\n".join(issues))
                        st.info("✅ Script will auto-fix these issues during processing")
                    
                    # Store all orders in session state (parsed once)
                    st.session_state.all_orders = df.to_dict('records')
                    st.session_state.total_orders = len(df)
                    st.session_state.file_uploaded = True
                    st.session_state.batch_id = str(uuid.uuid4()).replace("-", "")[:20]
                    st.session_state.carrier = "RS"
                    
                    st.success(f"✅ Loaded {st.session_state.total_orders} orders! Ready to process.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Parse error: {str(e)}")
    else:
        # FILE ALREADY LOADED - Show Progress & Continue Button
        processed = len(st.session_state.processed_indices)
        total = st.session_state.total_orders
        remaining = total - processed
        percent = (processed / total * 100) if total > 0 else 0
        
        st.success(f"✅ File loaded: {total} orders")
        st.progress(percent / 100)
        st.metric("Progress", f"{processed}/{total} orders ({percent:.1f}%)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            continue_disabled = remaining <= 0 or st.session_state.processing_complete
            if st.button("▶️ Process Next Batch", type="primary", disabled=continue_disabled, use_container_width=True):
                batch_start = processed
                batch_end = min(batch_start + batch_size, total)
                batch_indices = list(range(batch_start, batch_end))
                
                with st.spinner(f"⏳ Processing orders {batch_start + 1}-{batch_end} of {total}..."):
                    batch_results = []
                    for idx in batch_indices:
                        try:
                            row = st.session_state.all_orders[idx]
                            
                            # ✅ FIX: Use length/width/height instead of lwh (lwh has wrong data)
                            length = safe_float(row.get('length') or row.get('lwh', 10), 10)
                            width = safe_float(row.get('width') or row.get('lwh', 10), 10)
                            height = safe_float(row.get('height') or row.get('lwh', 10), 10)
                            
                            # ✅ FIX: If lwh values are unrealistic (>1000), use defaults
                            if length > 1000 or width > 1000 or height > 1000:
                                length = safe_float(row.get('length', 10), 10)
                                width = safe_float(row.get('width', 10), 10)
                                height = safe_float(row.get('height', 10), 10)
                            
                            # ✅ FIX: Handle missing city - use "Toronto" as default
                            city = str(row.get('city', '') or '').strip()
                            if not city:
                                city = "Toronto"  # Default city for Canadian addresses
                            
                            # ✅ FIX: Handle missing province
                            province = str(row.get('province', 'ON') or 'ON').strip().upper()
                            if len(province) > 2:
                                province = province[:2]  # Truncate to 2 chars
                            
                            # ✅ FIX: Handle missing country
                            country = str(row.get('country', 'CA') or 'CA').strip().upper()
                            if not country:
                                country = "CA"
                            
                            # ✅ FIX: Handle empty services for RS
                            service_level = str(row.get('services', '') or '').strip()
                            
                            order_packages = [{
                                "Weight": safe_float(row.get('weight', 1), 1.0),
                                "Length": length,
                                "Width": width,
                                "Height": height,
                                "PackagingWeight": safe_float(row.get('packaging_weight', 0), 0.0),
                                "SKU": str(row.get('sku', 'N/A')),
                                "Description": str(row.get('description', 'No description')),
                                "Address": {
                                    "Name": str(row.get('name', 'John Doe')),
                                    "Company": str(row.get('company', '')),
                                    "Address1": str(row.get('address', '')),
                                    "Address2": str(row.get('address2', '')),
                                    "City": city,  # ✅ Now has default value
                                    "StateProvince": province,
                                    "Postal": str(row.get('postal', '')).replace(" ", "").upper()[:10],
                                    "Country": country,
                                    "Phone": str(row.get('phone', '')),
                                    "Email": str(row.get('email', ''))
                                },
                                "ServiceLevel": service_level,
                                "Carrier": "RS",
                                "ClientCode": str(row.get('client_code', fallback_client_code)) or fallback_client_code,
                                "OrderID": str(row.get('order_id', f'ORD-{idx}'))[:20]
                            }]
                            
                            result = process_order_batch(order_packages, "RS", fallback_client_code.strip(),
                                                        st.session_state.batch_id, dry_run, chunk_size)
                            batch_results.append(result)
                            
                        except Exception as e:
                            batch_results.append({
                                "Status": "❌ Error", "OrderID": f"ORD-{idx}", "TransactionNumber": "N/A",
                                "BatchID": st.session_state.batch_id, "Boxes": 0, "Cost": "$0.00",
                                "BaseAmount": "$0.00", "FuelSurcharge": "$0.00", "PublicTotal": "$0.00",
                                "Service": "N/A", "Carrier": "N/A", "Recipient": "N/A", "PostalCode": "N/A",
                                "Chunks": "0/0", "Error": str(e)[:100], "DryRun": dry_run
                            })
                    
                    # Save results
                    st.session_state.all_results.extend(batch_results)
                    st.session_state.processed_indices.extend(batch_indices)
                    
                    # ✅ Show detailed error summary if all failed
                    if len(batch_results) > 0:
                        failed = sum(1 for r in batch_results if "❌" in r.get("Status", "") or "⚠️" in r.get("Status", ""))
                        if failed == len(batch_results):
                            st.error(f"❌ All {len(batch_results)} orders failed. Check errors below:")
                            for r in batch_results[:3]:
                                if r.get("Error"):
                                    st.write(f"- {r.get('OrderID')}: {r.get('Error')}")
                            st.info("💡 **Common fixes:** Check postal codes are valid, city/province filled, client code active")
                        else:
                            st.success(f"✅ Batch complete! Processed {len(batch_results)} orders. Total: {len(st.session_state.processed_indices)}/{total}")
                    
                    if len(st.session_state.processed_indices) >= total:
                        st.session_state.processing_complete = True
                        st.balloons()
                    
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.session_state.all_results:
                csv = pd.DataFrame(st.session_state.all_results).to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Results", csv, f"techship_batch_{st.session_state.batch_id}.csv", "text/csv", use_container_width=True)
        
        with col4:
            if st.button("🗑️ Reset & New File", use_container_width=True):
                for key in ["file_uploaded", "all_orders", "processed_indices", "all_results", "batch_id", "carrier", "total_orders", "processing_complete"]:
                    st.session_state[key] = [] if key in ["all_orders", "processed_indices", "all_results"] else "" if key == "batch_id" or key == "carrier" else False if key == "file_uploaded" or key == "processing_complete" else 0 if key == "total_orders" else None
                st.rerun()
        
        # DISPLAY RESULTS (Always Visible)
        if st.session_state.all_results:
            st.subheader(f"📊 Results So Far ({len(st.session_state.all_results)} orders)")
            
            results_df = pd.DataFrame(st.session_state.all_results)
            display_cols = ["OrderID", "Status", "Boxes", "Cost", "Service", "Recipient", "PostalCode", "Chunks"]
            if "Error" in results_df.columns:
                display_cols.append("Error")
            
            st.dataframe(results_df[display_cols], use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            success = sum(1 for r in st.session_state.all_results if "✅" in r.get("Status", ""))
            col1.metric("Successful", success)
            col2.metric("Failed", len(st.session_state.all_results) - success)
            col3.metric("Total Cost", f"${sum(safe_float(r.get('Cost', '$0').replace('$', '')) for r in st.session_state.all_results):.2f}")
        
        # SHOW REMAINING INFO
        if remaining > 0:
            st.info(f"⏭️ {remaining} orders remaining. Click **Process Next Batch** to continue.")
            est_time = (remaining / batch_size) * 1
            st.write(f"⏱️ Estimated time remaining: ~{est_time:.0f} minutes")
        else:
            st.success("🎉 All orders processed!")

if __name__ == "__main__":
    main()
