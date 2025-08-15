# ==============================================

# SAFE EDITING RULES - READ BEFORE MAKING CHANGES

# ==============================================

# 1. Use 4 spaces for indentation. Do NOT use tabs.

# 2. Keep code that uses local variables inside the same function.

# 3. After 'def/if/with/try/for', the next line must be indented.

# 4. Place error logs using 'e' only inside 'except Exception as e:' blocks.

# 5. Prefer commenting out large blocks instead of deleting during tests.

# 6. Test after each small change (save → run) to catch errors early.

# ==============================================



import streamlit as st
import openai
import json
import re
import requests
import os
import time
import base64
from openai import OpenAI
import uuid
import datetime as dt
from supabase import create_client, Client
import bcrypt
from pathlib import Path

# ---- Load system prompts from external markdown files ----
@st.cache_data
def load_prompt(file_path):
    """Load a text/markdown prompt file into a string."""
    return Path(file_path).read_text(encoding="utf-8")

TEXT_PROMPT_PATH = Path(__file__).parent / "prompts" / "text_system.md"
IMAGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "image_system.md"

TEXT_SYSTEM_PROMPT = load_prompt(TEXT_PROMPT_PATH)
IMAGE_SYSTEM_PROMPT = load_prompt(IMAGE_PROMPT_PATH)

# Initialize Image-Generator session state
if "img_mode" not in st.session_state:
    st.session_state.img_mode = "Create"
if "chained_image" not in st.session_state:
    st.session_state.chained_image = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = None

# ---- JSON handling functions from content builder ----
def extract_first_json(text):
    """
    Improved JSON extraction with better error handling
    """
    text = text.strip()
    
    # Handle array format
    if text.startswith("["):
        try:
            arr = json.loads(text)
            return arr[0] if arr else {}
        except json.JSONDecodeError as e:
            st.error(f"JSON Array Parse Error: {e}")
            return create_fallback_response()
    
    # Handle single object format
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            st.error(f"JSON Object Parse Error: {e}")
            return create_fallback_response()
    
    # Try to extract JSON from mixed content
    try:
        # Look for JSON objects in the text
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return fallback
        return create_fallback_response()
        
    except Exception as e:
        st.error(f"JSON Extraction Error: {e}")
        return create_fallback_response()

def create_fallback_response():
    """
    Create a safe fallback response when JSON parsing fails
    """
    return {
        "body": "Sorry, I can only provide campaign content for business messaging. Please revise your prompt.",
        "placeholders": [],
        "length": 88,
        "variant_id": None
    }

def sanitize_json_string(text):
    """
    Sanitize strings to prevent JSON parsing issues
    """
    if not isinstance(text, str):
        return text
    
    # Escape problematic characters
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace('\n', '\\n')   # Escape newlines
    text = text.replace('\r', '\\r')   # Escape carriage returns
    text = text.replace('\t', '\\t')   # Escape tabs
    
    return text

def safe_json_dumps(obj):
    """
    Safely convert object to JSON string with error handling
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        st.error(f"JSON Serialization Error: {e}")
        return json.dumps(create_fallback_response(), indent=2)

def unescape_json_string(text):
    """
    Unescape JSON string for display purposes
    """
    if not isinstance(text, str):
        return text
    
    # Unescape common JSON escape sequences
    text = text.replace('\\"', '"')    # Unescape double quotes
    text = text.replace('\\\\', '\\')  # Unescape backslashes
    text = text.replace('\\n', '\n')   # Unescape newlines
    text = text.replace('\\r', '\r')   # Unescape carriage returns
    text = text.replace('\\t', '\t')   # Unescape tabs
    
    return text

def validate_and_fix_output(output_dict):
    """
    Validate and fix common issues in AI output
    """
    # Ensure required fields exist
    required_fields = ["body", "placeholders", "length", "variant_id"]
    for field in required_fields:
        if field not in output_dict:
            if field == "body":
                output_dict[field] = "Content generation error"
            elif field == "placeholders":
                output_dict[field] = []
            elif field == "length":
                output_dict[field] = len(output_dict.get("body", ""))
            elif field == "variant_id":
                output_dict[field] = None
    
    # Don't sanitize here - let the content be natural for display
    # Sanitization will happen when we convert to JSON for storage
    
    # Ensure placeholders is a list
    if not isinstance(output_dict.get("placeholders"), list):
        output_dict["placeholders"] = []
    
    # Ensure length is a number
    if not isinstance(output_dict.get("length"), (int, float)):
        output_dict["length"] = len(output_dict.get("body", ""))
    
    return output_dict

# ---- Image generation functions ----
FLUX_SYSTEM_PROMPT = IMAGE_SYSTEM_PROMPT

# Image Gen Helper Functions

def enhance_prompt(raw_prompt: str) -> str:
    """Call GPT-4o-mini to refine the raw prompt."""
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FLUX_SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def generate_flux(prompt: str) -> bytes:
    """Call Replicate Flux Schnell API and return image bytes."""
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt
        }
    }
    
    api_endpoint = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
    
    try:
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()
        prediction_id = prediction["id"]
        
        # Poll until completion with timeout
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            
            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                if isinstance(outputs, list) and len(outputs) > 0:
                    image_url = outputs[0]
                elif isinstance(outputs, str):
                    image_url = outputs
                else:
                    raise Exception("No valid output URL found")
                
                # Download the image
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return img_resp.content
                
            elif status_data["status"] == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise Exception(f"Prediction failed: {error_msg}")
            
            time.sleep(2)  # Wait 2 seconds before next poll
        
        raise Exception("Generation timed out after 5 minutes")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Replicate API request error: {str(e)}")
    except Exception as e:
        raise Exception(f"Image generation error: {str(e)}")
        
def generate_kontext_max(prompt: str, input_image_uri: str) -> bytes:
    """Call Replicate Flux Kontext Max API and return image bytes."""
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "input_image": input_image_uri,
            "output_format": "jpg",
        }
    }
    api_endpoint = "https://api.replicate.com/v1/models/black-forest-labs/flux-kontext-max/predictions"

    try:
        # kick off the prediction
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()
        prediction_id = prediction["id"]

        # Poll until completion (same timeout as generate_flux)
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                # handle list or single URL
                url = outputs[0] if isinstance(outputs, list) else outputs
                img_resp = requests.get(url)
                img_resp.raise_for_status()
                return img_resp.content

            if status_data["status"] == "failed":
                err = status_data.get("error", "Unknown error")
                raise Exception(f"Prediction failed: {err}")

            time.sleep(2)

        raise Exception("Generation timed out after 5 minutes")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Replicate API request error: {e}")
    except Exception as e:
        raise Exception(f"Image generation error: {e}")
        
        
#Multi Image Kontext Helper Function
def generate_multi_image_kontext_base64(
    prompt: str,
    image_files,
    aspect_ratio: str = "match_input_image",
    model_slug: str = "flux-kontext-apps/multi-image-list",
) -> bytes:
    """
    Alternative implementation using base64 data URLs instead of file uploads
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")
    if not image_files or len(image_files) == 0:
        raise ValueError("At least one input image is required.")

    token = st.secrets["REPLICATE_API_TOKEN"]
    
    try:
        # Convert images to base64 data URLs
        image_data_urls = []
        
        for i, f in enumerate(image_files[:4]):
            try:
                if hasattr(f, "seek"):
                    f.seek(0)
            except Exception:
                pass
            
            # Read file data
            if hasattr(f, 'read'):
                file_data = f.read()
                if hasattr(f, 'seek'):
                    f.seek(0)
            else:
                file_data = f
            
            # Get content type
            content_type = getattr(f, 'type', 'image/png')
            
            # Convert to base64 data URL
            b64_data = base64.b64encode(file_data).decode('utf-8')
            data_url = f"data:{content_type};base64,{b64_data}"
            image_data_urls.append(data_url)
        
        # Create prediction payload
        payload = {
            "input": {
                "prompt": prompt.strip(),
                "input_images": image_data_urls,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "safety_tolerance": 2
            }
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "wait"
        }
        
        api_endpoint = f"https://api.replicate.com/v1/models/{model_slug}/predictions"
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()
        
        # Handle response (same polling logic as above)
        prediction_id = prediction["id"]
        max_wait_time = 300
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            
            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                image_url = outputs[0] if isinstance(outputs, list) else outputs
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return img_resp.content
                
            elif status_data["status"] == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise Exception(f"Prediction failed: {error_msg}")
            
            time.sleep(2)
        
        raise Exception("Generation timed out after 5 minutes")
        
    except Exception as e:
        raise Exception(f"Multi-image generation error: {str(e)}")



# ---- Page configuration and styling ----
st.set_page_config(page_title="AI Content & Image Generator", layout="centered")
GMS_TEAL = "#E6F9F3"
GMS_GREEN = "#22B573"
GMS_BLUE = "#C7E7FD"
GMS_LAVENDER = "#D5D7FB"

# ---- Custom CSS for GMS color palette and rounded corners ----
st.markdown(f"""
    <style>
        .stApp {{ background-color: {GMS_TEAL}; }}
        /* Title styling - outside the block container */
        .page-title {{
            color: {GMS_GREEN};
            text-align: center;
            margin-top: 40px;
            margin-bottom: 1.5em;
            font-size: 2.5rem;
            font-weight: 700;
            position: relative;
            z-index: 1000;
        }}
    </style>
""", unsafe_allow_html=True)

# ---- Logo positioned at top left ----
if os.path.exists("gms_logo.png"):
    st.markdown(
        """
        <div style='position: fixed; top: 80px; left: 20px; z-index: 999;'>
            <img src='data:image/png;base64,{}' width='250'>
        </div>
        """.format(
            base64.b64encode(open("gms_logo.png", "rb").read()).decode()
        ), 
        unsafe_allow_html=True
    )
else:
    # Fallback if logo file doesn't exist
    st.markdown(f"<div style='position: fixed; top: 20px; left: 20px; z-index: 999; color: {GMS_GREEN}; font-size: 1.2rem; font-weight: 700;'>●●● gms</div>", unsafe_allow_html=True)

# ---- Title positioned outside the form ----
st.markdown(f"<h1 class='page-title'>AI Content Builder</h1>", unsafe_allow_html=True)

# ---- Form container CSS ----
st.markdown(f"""
    <style>
        .block-container {{
            background-color: white !important;
            border-radius: 24px;
            padding: 2em 3em;
            margin-top: 0em;
            box-shadow: 0 0 20px {GMS_LAVENDER};
        }}
        .stButton>button, button[kind="primary"] {{
            background-color: {GMS_GREEN} !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            margin: 0.25em 0.5em 0.25em 0 !important;
        }}
        .stButton>button:hover {{
            background-color: #19995a !important;
            color: white !important;
        }}
        .stTextInput>div>div>input,
        .stTextArea textarea {{
            background-color: {GMS_BLUE}10 !important;
            border-radius: 8px;
        }}
        .error-message {{
            background-color: #ffebee;
            color: #c62828;
            padding: 1em;
            border-radius: 8px;
            margin: 1em 0;
        }}
        /* Tab styling - centered and 50% each */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0px;
            background-color: {GMS_BLUE}20;
            border-radius: 12px;
            padding: 4px;
            display: flex;
            justify-content: center;
            max-width: 600px;
            margin: 0 auto;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            border-radius: 8px;
            font-weight: 600;
            background-color: transparent;
            flex: 1;
            text-align: center;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {GMS_GREEN} !important;
            color: white !important;
        }}
        /* Success/Error message styling */
        .stSuccess {{
            border-radius: 8px !important;
            border-left: 4px solid {GMS_GREEN} !important;
        }}
        .stError {{
            border-radius: 8px !important;
            border-left: 4px solid #dc3545 !important;
        }}
        /* Image display styling */
        .generated-image {{
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            margin: 1.5rem 0;
        }}
        /* Download button styling */
        .stDownloadButton > button {{
            background-color: #6c757d !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            height: 3rem !important;
            width: 100% !important;
            margin-top: 1rem !important;
        }}
        .stDownloadButton > button:hover {{
            background-color: #5a6268 !important;
            transform: translateY(-1px) !important;
        }}
        /* Magic image at top right */
        .magic-image {{
            position: fixed;
            bottom: 80px;
            right: 20px;
            z-index: 999;
            width: 200px;
            opacity: 0.8;
        }}
    </style>
""", unsafe_allow_html=True)

# ---- Magic image at bottom right ----
if os.path.exists("magic.png"):
    st.markdown(
        """
        <div class='magic-image'>
            <img src='data:image/png;base64,{}' width='100'>
        </div>
        """.format(
            base64.b64encode(open("magic.png", "rb").read()).decode()
        ), 
        unsafe_allow_html=True
    )

# ---- Content generation system prompt ----
system_prompt = TEXT_SYSTEM_PROMPT

# ---- Initialize session state ----
# Content generation state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt}
    ]
if "raw_input_text" not in st.session_state:
    st.session_state.raw_input_text = ""
if "raw_output_text" not in st.session_state:
    st.session_state.raw_output_text = ""
if "last_output" not in st.session_state:
    st.session_state.last_output = None
if "last_variants" not in st.session_state:
    st.session_state.last_variants = []
if "selected_variant" not in st.session_state:
    st.session_state.selected_variant = 0

# Image generation state
if "refined_prompt" not in st.session_state:
    st.session_state.refined_prompt = ""


# ---- Supabase + Auth/Analytics Helpers ----
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_SERVICE_ROLE")
sb: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def _get_user_by_username(username: str):
    if not sb: return None
    resp = sb.table("users").select("id, username, password_hash, role").eq("username", username).limit(1).execute()
    rows = resp.data or []
    return rows[0] if rows else None

def _verify_password(plain: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), pw_hash.encode())
    except Exception:
        return False

def _bootstrap_admin_if_needed():
    if not sb: return
    try:
        existing = sb.table("users").select("id").eq("role", "admin").limit(1).execute().data or []
        if existing: return
        uname = st.secrets.get("BOOTSTRAP_ADMIN_USERNAME")
        pwd = st.secrets.get("BOOTSTRAP_ADMIN_PASSWORD")
        if uname and pwd:
            pw_hash = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
            sb.table("users").insert({"username": uname, "password_hash": pw_hash, "role": "admin"}).execute()
    except Exception as e:
        st.warning(f"Admin bootstrap skipped: {e}")

def log_event(user_id: str | None, action: str, **props):
    if not sb: return
    data = {"action": action, **props}
    if user_id: data["user_id"] = user_id
    sb.table("events").insert(data).execute()

def save_image_return_url(user_id: str, img_bytes: bytes, mode: str, model: str, prompt: str):
    if not sb:
        return None, None
    image_id = str(uuid.uuid4())
    path = f"images/{image_id}.png"
    sb.storage.from_("images").upload(path, img_bytes, {"content-type": "image/png"})
    url = sb.storage.from_("images").get_public_url(path)
    row = {"id": image_id, "user_id": user_id, "url": url, "bytes": len(img_bytes), "mode": mode, "model": model, "prompt": prompt}
    sb.table("images").insert(row).execute()
    return image_id, url

# ==============================================
# ADMIN HELPER: count_success_images_today — Ensure it excludes meta.voided=true
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================

def count_success_images_today(user_id: str) -> int:
    if not sb: return 0
    now = dt.datetime.utcnow()
    start = dt.datetime(now.year, now.month, now.day).isoformat()
    end = (dt.datetime(now.year, now.month, now.day) + dt.timedelta(days=1)).isoformat()
    excluded = _get_excluded_quota_modes()
    q = (sb.table("events")
            .select("id, mode, meta", count="exact")
            .eq("user_id", user_id)
            .eq("action", "image_generate_done")
            .eq("status", "success")
            .gte("ts", start).lt("ts", end)
            .or_("meta->>voided.is.null,meta->>voided.eq.false"))
    if excluded:
        q = q.not_.in_("mode", list(excluded))
    resp = q.execute()
    return resp.count or 0
    
# ==============================================
# ADMIN HELPER: This function helps removing any particular image mode from 
# user's daily quota count. For example, you may remove pillow related usecases from quota in future
# ==============================================
    
def _get_excluded_quota_modes():
    raw = st.secrets.get("EXCLUDE_FROM_QUOTA_MODES", "").strip()
    return set([m.strip() for m in raw.split(",") if m.strip()])  # e.g., {"text_overlay","resize_crop"}

# ==============================================
# ADMIN HELPER: reset_today_quota — Marks events as meta.voided; never delete
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
def reset_today_quota(user_id: str):
    if not sb: 
        return 0
    now = dt.datetime.utcnow()
    start = dt.datetime(now.year, now.month, now.day).isoformat()
    end = (dt.datetime(now.year, now.month, now.day) + dt.timedelta(days=1)).isoformat()
    # Fetch today's success events
    rows = (
        sb.table("events")
          .select("id, meta")
          .eq("user_id", user_id)
          .eq("action", "image_generate_done")
          .eq("status", "success")
          .gte("ts", start)
          .lt("ts", end)
          .execute()
    ).data or []
    updated = 0
    for r in rows:
        meta = r.get("meta") or {}
        if meta.get("voided") is True:
            continue
        meta["voided"] = True
        sb.table("events").update({"meta": meta}).eq("id", r["id"]).execute()
        updated += 1
    return updated

def list_users():
    if not sb: return []
    return sb.table("users").select("id, username, role").order("username").execute().data or []

# ==============================================
# ADMIN HELPER: fetch_events — Safe to edit filters/selection; keep return shape the same
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
def fetch_events(start_iso: str, end_iso: str, user_id: str | None, mode: str | None):
    if not sb: 
        return []
    q = (
        sb.table("events")
          .select("id, ts, user_id, action, mode, model, status, image_id, prompt, meta")
          .gte("ts", start_iso).lt("ts", end_iso)
          .in_("action", ["text_generate_done","image_generate_done","image_refine_prompt"])
          .order("ts", desc=True)
    )
    if user_id:
        q = q.eq("user_id", user_id)
    if mode and mode != "All":
        q = q.eq("mode", mode)
    return q.execute().data or []

def ensure_auth_state():
    if 'auth_user' not in st.session_state:
        st.session_state.auth_user = None

def render_login():
    _bootstrap_admin_if_needed()
    st.title("🔐 Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        u = _get_user_by_username(username.strip())
        if not u or not _verify_password(password, u["password_hash"]):
            st.error("Invalid credentials.")
            st.stop()
        st.session_state.auth_user = {"id": u["id"], "username": u["username"], "role": u["role"]}
        st.success(f"Welcome, {u['username']}!")
        st.rerun()

# ==============================================
# ADMIN UI START — Everything below stays INSIDE this function; paste UI here
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
def render_admin():
    st.title("📊 Admin")

    # ---- Filters ----
    today = dt.date.today()
    colf1, colf2, colf3 = st.columns([1, 1, 1])
    with colf1:
        start = st.date_input("From", today - dt.timedelta(days=7))
    with colf2:
        end = st.date_input("To", today + dt.timedelta(days=1))

    users = list_users()
    user_map = {u["username"]: u["id"] for u in users}
    with colf3:
        user_choice = st.selectbox("User", ["All"] + list(user_map.keys()))
    mode_choice = st.selectbox(
        "Mode",
        ["All", "text", "create", "inspire", "combine", "remove_bg", "text_overlay", "color_swap", "resize_crop"]
    )

    start_iso = dt.datetime.combine(start, dt.time.min).isoformat()
    end_iso = dt.datetime.combine(end, dt.time.min).isoformat()
    uid = None if user_choice == "All" else user_map[user_choice]
    rows = fetch_events(start_iso, end_iso, uid, None if mode_choice == "All" else mode_choice)

    # ---- Usage today / quota ----
    st.markdown("#### Usage today")
    if uid:
        used = count_success_images_today(uid)
        st.write(f"Images today: **{used} / 10**")
        if st.button("Reset today's image quota for this user"):
            removed = reset_today_quota(uid)
            st.success(f"Reset done. Voided {removed} success events.")
    else:
        st.info("Select a user to view/reset quota.")

    # ---- Quick stats (lightweight) ----
    st.markdown("#### Quick stats")
    try:
        _now = dt.datetime.utcnow()
        _start = dt.datetime(_now.year, _now.month, _now.day)
        _end = _start + dt.timedelta(days=1)

        _img_ok = (
            sb.table("events").select("id", count="exact")
            .eq("action", "image_generate_done").eq("status", "success")
            .or_("meta->>voided.is.null,meta->>voided.eq.false")
            .gte("ts", _start.isoformat()).lt("ts", _end.isoformat())
            .execute()
        ).count or 0

        _txt_ok = (
            sb.table("events").select("id", count="exact")
            .eq("action", "text_generate_done").eq("status", "success")
            .gte("ts", _start.isoformat()).lt("ts", _end.isoformat())
            .execute()
        ).count or 0

        _errs = (
            sb.table("events").select("id", count="exact")
            .in_("action", ["text_generate_done", "image_generate_done"])
            .eq("status", "error")
            .gte("ts", _start.isoformat()).lt("ts", _end.isoformat())
            .execute()
        ).count or 0

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Image successes (today)", _img_ok)
        with c2: st.metric("Text successes (today)", _txt_ok)
        with c3:
            _tot = _img_ok + _txt_ok + _errs
            _rate = (100.0 * (_img_ok + _txt_ok) / _tot) if _tot else 0.0
            st.metric("Success rate (today)", f"{_rate:.0f}%")

        _week_start = (_now - dt.timedelta(days=7)).isoformat()
        _img_rows = (
            sb.table("events").select("mode")
            .eq("action", "image_generate_done").eq("status", "success")
            .gte("ts", _week_start).execute()
        ).data or []
        _mode_counts = {}
        for r in _img_rows:
            _m = r.get("mode") or "unknown"
            _mode_counts[_m] = _mode_counts.get(_m, 0) + 1
        if _mode_counts:
            st.bar_chart(_mode_counts, use_container_width=True)
    except Exception as _e:
        st.info(f"Stats unavailable: {_e}")

    # ---- Events table ----
    st.markdown("#### Events")
    if rows:
        uname_lookup = {u["id"]: u["username"] for u in users}
        from zoneinfo import ZoneInfo
        _tz = ZoneInfo("Asia/Karachi")

        def _short(s, n=140):
            s = s or ""
            return (s[:n] + "…") if len(s) > n else s

        table = []
        for r in rows:
            try:
                ts_raw = r["ts"]
                if isinstance(ts_raw, str):
                    _dt = dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                else:
                    _dt = ts_raw
                if getattr(_dt, "tzinfo", None) is None:
                    _dt = _dt.replace(tzinfo=dt.timezone.utc)
                date_str = _dt.astimezone(_tz).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_str = str(r["ts"]).replace("T", " ").split(".")[0]

            uname = uname_lookup.get(r["user_id"], "—")
            action = r.get("action")
            result = (
                "Refine" if action == "image_refine_prompt"
                else ("Error" if r.get("status") == "error"
                      else ("Image" if r.get("image_id") else "Text"))
            )

            meta = r.get("meta") or {}
            if result == "Error":
                details = _short(meta.get("error", ""))
            elif result == "Text":
                details = _short(meta.get("text", ""))
            elif result == "Image":
                details = meta.get("file") or meta.get("url") or ""
            elif result == "Refine":
                details = _short(meta.get("after", ""))
            else:
                details = ""

            table.append({
                "Date": date_str,
                "User": uname,
                "Mode": r.get("mode"),
                "Prompt": _short(r.get("prompt") or ""),
                "Result": result,
                "Details": details,
            })

        st.dataframe(table, use_container_width=True, hide_index=True)

        try:
            row_idx = st.number_input(
                "View row #", min_value=0, max_value=max(0, len(rows) - 1), value=0
            )
            st.text_area("Full Prompt", rows[row_idx].get("prompt") or "", height=140)
            _m = rows[row_idx].get("meta") or {}
            st.text_area(
                "Full Details",
                (_m.get("error") or _m.get("text") or _m.get("after") or _m.get("url") or ""),
                height=180,
            )
        except Exception:
            pass
    else:
        st.info("No events for the selected filters.")

    # ---- Recent images gallery ----
    st.markdown("#### Recent images")
    try:
        imgs = (
            sb.table("images")
            .select("id, url, user_id, created_at, mode")
            .order("created_at", desc=True)
            .limit(30)
            .execute()
            .data
            or []
        )
    except Exception:
        imgs = []

    if imgs:
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            with cols[i % 3]:
                st.image(
                    img["url"],
                    caption=f"{img['mode']} • {str(img['created_at'])[:19]}",
                    use_container_width=True,
                )
                try:
                    data = requests.get(img["url"]).content
                    st.download_button("Download", data=data, file_name=f"{img['id']}.png")
                except Exception:
                    st.caption("Download failed.")
    else:
        st.info("No images yet.")

    # ---- Purge older than 30d ----
    st.divider()
    if st.button("🧹 Purge data older than 30 days"):
        ni, ne = purge_older_than_30d()
        st.success(f"Purged {ni} images and {ne} events.")

# ---- Main tab interface ----

# ---- Auth Gate ----
ensure_auth_state()
if not st.session_state.auth_user:
    render_login()
    st.stop()

# If admin, show only admin and exit
if st.session_state.auth_user.get("role") == "admin":
    render_admin()
    st.stop()

tab1, tab2 = st.tabs(["📝 Text Generator", "🎨 Image Generator"])

# ---- CONTENT GENERATOR TAB ----
# ==============================================
# TEXT TAB START — Keep OpenAI call & success/error logs; add UI only
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
with tab1:
    # ---- Input Form ----
    with st.form("campaign_form"):
        st.subheader("Campaign Details")
        channel = st.selectbox("Channel", ["whatsapp", "sms", "viber"])
        prompt = st.text_area(
            "Campaign Instruction / Prompt",
            placeholder="Describe your campaign, product details, offer, and any special instructions."
        )
        language = st.text_input("Language", "en")
        tone = st.text_input("Tone", "friendly")
        max_length = st.number_input("Max Length", min_value=1, max_value=1024, value=250)
        variants = st.number_input("Number of Variants", min_value=1, max_value=3, value=1)
        generate_btn = st.form_submit_button("Generate Content")

    # ---- GENERATE CONTENT: starts a NEW session ----
    if generate_btn and prompt:
        # Log text generate click (does not count toward quota)
        try:
            # ==============================================
            # LOGGING: log_event — If moving, keep inside correct try/except and scope
            # (Non-executable guidance. Do not remove. Keep indentation consistent.)
            # ==============================================
            log_event(st.session_state.auth_user["id"], "text_generate_click", mode="text", prompt=prompt, prompt_len=len(prompt))
        except Exception:
            pass

        openai_api_key = st.secrets["OPENAI_API_KEY"]
        client = openai.OpenAI(api_key=openai_api_key)

        # Reset chat history to only system prompt (new session)
        st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

        input_json = {
            "prompt": prompt,
            "channel": channel,
            "language": language,
            "tone": tone,
            "maxLength": max_length,
            "variants": int(variants)
        }

        # Add the new user message -- always valid JSON!
        try:
            st.session_state.chat_history.append(
                {"role": "user", "content": safe_json_dumps(input_json)}
            )
        except Exception as e:
            st.error(f"Error preparing request: {e}")
            st.stop()

        try:
            # ==============================================
            # CORE CALL: OpenAI completion — Do not modify request/response handling
            # (Non-executable guidance. Do not remove. Keep indentation consistent.)
            # ==============================================
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.chat_history,
                max_tokens=2000,
                temperature=0.7,
                n=int(variants)
            )
            
            # Collect variants with robust JSON extraction
            variant_list = []
            for i in range(int(variants)):
                output = response.choices[i].message.content.strip()
                
                # Debug: Show raw output in expander
                with st.expander(f"Debug: Raw GPT Output for Variant {i+1}"):
                    st.text(output)
                
                try:
                    # Try direct JSON parsing first
                    if output.startswith('['):
                        arr = json.loads(output)
                        result = arr[i] if i < len(arr) else arr[0] if arr else create_fallback_response()
                    else:
                        result = json.loads(output)
                    
                    # Validate and fix the result
                    result = validate_and_fix_output(result)
                    
                except json.JSONDecodeError as e:
                    st.warning(f"JSON parsing failed for variant {i+1}: {e}")
                    try:
                        result = extract_first_json(output)
                        result = validate_and_fix_output(result)
                    except Exception as e2:
                        st.error(f"Fallback JSON extraction failed: {e2}")
                        result = create_fallback_response()
                
                except Exception as e:
                    st.error(f"Unexpected error processing variant {i+1}: {e}")
                    result = create_fallback_response()
                
                variant_list.append(result)

            st.session_state.last_variants = variant_list
            st.session_state.selected_variant = 0
            st.session_state.last_output = variant_list[0]

            # ---- Reset chat_history to just system + user + assistant (of selected variant) ----
            st.session_state.chat_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_json_dumps(input_json)},
                {"role": "assistant", "content": safe_json_dumps(st.session_state.last_output)}
            ]

            # ---- Store RAW INPUT and RAW OUTPUT for always-visible debug ----
            st.session_state.raw_input_text = safe_json_dumps(st.session_state.chat_history)
            st.session_state.raw_output_text = safe_json_dumps(st.session_state.last_output)

            st.success("Content generated successfully!")

            # Store text in meta for admin Details
            try:
                # ==============================================
                # LOGGING: log_event — If moving, keep inside correct try/except and scope
                # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                # ==============================================
                log_event(st.session_state.auth_user["id"], "text_generate_done", mode="text", status="success", prompt=prompt, prompt_len=len(prompt), meta={"text": (st.session_state.last_output or {}).get("body","")})
            except Exception:
                pass
                
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")

        try:
            # ==============================================
            # LOGGING: log_event — If moving, keep inside correct try/except and scope
            # (Non-executable guidance. Do not remove. Keep indentation consistent.)
            # ==============================================
            log_event(st.session_state.auth_user["id"], "text_generate_done", mode="text", status="error", prompt=prompt, prompt_len=len(prompt), meta={"error": str(e)})
        except Exception:
            pass

    # ---- Variant selector if multiple ----
    if st.session_state.last_variants:
        if len(st.session_state.last_variants) > 1:
            options = [f"Variant {i+1}" for i in range(len(st.session_state.last_variants))]
            selected = st.selectbox("Select Variant to View/Edit", options,
                                    index=st.session_state.selected_variant)
            idx = options.index(selected)
            st.session_state.last_output = st.session_state.last_variants[idx]
            st.session_state.selected_variant = idx

            # ---- Update chat_history to reflect newly selected variant ----
            if "chat_history" in st.session_state and st.session_state.chat_history:
                if (len(st.session_state.chat_history) == 3 and 
                    st.session_state.chat_history[2]["role"] == "assistant"):
                    st.session_state.chat_history[2]["content"] = safe_json_dumps(st.session_state.last_output)

    # ---- OUTPUT section: Body + Placeholders Only ----
    if st.session_state.last_output:
        output = st.session_state.last_output
        st.markdown("### Generated Content")
        
        # Unescape the body content for display
        display_body = unescape_json_string(output.get("body", ""))
        body = st.text_area("Body", display_body, height=120, key="body_out")
        
        length = st.text_input("Length", str(output.get("length", "")), key="length_out", disabled=True)
        variant_id = st.text_input("Variant ID", output.get("variant_id", ""), key="variant_id_out", disabled=True)
        placeholders = output.get("placeholders", [])
        if placeholders:
            st.markdown(f"**Placeholders:** {', '.join(placeholders)}")

        st.markdown("---")
        st.markdown("#### Follow-up Prompt (for edits)")
        follow_up = st.text_input("Describe your change or revision", key="followup")
        edit_btn = st.button("Edit Content")

        # ---- EDIT CONTENT: continue the existing session ----
        if edit_btn and follow_up:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            client = openai.OpenAI(api_key=openai_api_key)

            try:
                # Get JSON-encoded user message and previous assistant message from chat_history
                base_user_content = st.session_state.chat_history[1]["content"]
                previous_output_content = st.session_state.chat_history[2]["content"]

                followup_message = {
                    "role": "user",
                    "content": safe_json_dumps({
                        "edit_instruction": follow_up,
                        "base_campaign": json.loads(base_user_content),
                        "previous_output": json.loads(previous_output_content)
                    })
                }
                st.session_state.chat_history.append(followup_message)

                # ==============================================
                # CORE CALL: OpenAI completion — Do not modify request/response handling
                # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                # ==============================================
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_history,
                    max_tokens=2000,
                    temperature=0.7,
                )
                
                output_text = response.choices[0].message.content
                
                # Debug: Show raw edit output
                with st.expander("Debug: Raw Edit Output"):
                    st.text(output_text)
                
                try:
                    result = extract_first_json(output_text)
                    result = validate_and_fix_output(result)
                except Exception as e:
                    st.error(f"Error parsing edit response: {e}")
                    result = create_fallback_response()

                # Append assistant response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": safe_json_dumps(result)}
                )

                st.session_state.last_output = result
                if st.session_state.last_variants:
                    idx = st.session_state.selected_variant
                    st.session_state.last_variants[idx] = result

                # ---- Store RAW INPUT and RAW OUTPUT for always-visible debug ----
                st.session_state.raw_input_text = safe_json_dumps(st.session_state.chat_history)
                st.session_state.raw_output_text = safe_json_dumps(result)

                st.success("Content edited successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Edit Error: {e}")
                # Show more detailed error information
                st.error(f"Error details: {str(e)}")

# ---- IMAGE GENERATOR TAB ----
# ==============================================
# IMAGE TAB START — Keep Replicate/Flux + upload; add UI only
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
with tab2:
    st.subheader("Image Generation Details")

    # Modes: Create (text->image), Inspire (style copy from single template), Combine Images (multi-image model)
    mode = st.selectbox("Mode", ["Create", "Inspire", "Combine Images"], key="img_mode")

    # ---------- CREATE ----------
    if mode == "Create":
        raw_prompt = st.text_input(
            "Enter your prompt",
            placeholder="Describe the image you want to generate...",
            key="image_raw_prompt",
        )

        current_refined = st.session_state.get("refined_prompt", "")
        editable_prompt = st.text_area(
            "Refined prompt (editable)",
            value=current_refined,
            placeholder="Short of ideas? Use the refine prompt option to get a more descriptive prompt.",
            height=120,
            key="image_editable_prompt",
        )
        if editable_prompt != current_refined:
            st.session_state.refined_prompt = editable_prompt

    # ---------- INSPIRE (single image to copy style) ----------
    elif mode == "Inspire":
        # show chained output if user clicked “Edit This Image” earlier
        if st.session_state.get("chained_image") and st.session_state.get("edit_mode") == mode:
            input_bytes = st.session_state.chained_image
            input_mime = "image/png"
            st.image(input_bytes, caption="Using previous output", use_container_width=True)
        else:
            uploaded = st.file_uploader(
                "Upload an image to copy style from",
                type=["png", "jpg", "jpeg", "webp"],
                key="input_image_file_inspire",
            )
            if uploaded:
                input_bytes = uploaded.read()
                input_mime = uploaded.type
                st.image(input_bytes, caption="Uploaded image", use_container_width=True)
            else:
                input_bytes, input_mime = None, None

        prompt_inspire = st.text_input("Enter your prompt", key="img_prompt_inspire")

    # ---------- COMBINE IMAGES (multi‑image kontext) ----------
    else:
        st.caption("Upload up to 4 images. The model will combine/transform them per your prompt.")

        # If chaining from previous output, show it and allow up to 3 more uploads
        prefilled = st.session_state.get("chained_image") if st.session_state.get("edit_mode") == "Combine Images" else None
        if prefilled:
            st.image(prefilled, caption="Using previous output (counts as 1 image)", use_container_width=True)

        multi_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True,
            key="input_images_combine",
        )

        prompt_combine = st.text_input(
            "Enter your prompt",
            key="img_prompt_combine",
            placeholder="e.g., Put the product from image 1 on the background of image 2 with a summer vibe",
        )

        aspect = st.selectbox(
            "Aspect ratio",
            ["match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9", "9:21", "2:1", "1:2"],
            index=0,
            key="combine_aspect",
        )

    # ---------- Reset & Refine / Generate buttons ----------
    col1, col2 = st.columns([1, 1])

    # Reset All
    with col1:
        if st.button("🔄 Reset All", key="reset_all", use_container_width=True):
            for k in [
                "image_raw_prompt", "refined_prompt", "chained_image", "edit_mode",
                "img_prompt_inspire", "img_prompt_combine", "img_mode", "combine_aspect"
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        # Refine only in Create
        if mode == "Create" and st.button("🔄 Refine Prompt", key="refine_prompt_btn", use_container_width=True):
            if not raw_prompt or not raw_prompt.strip():
                st.error("❌ Please enter a prompt to refine.")
            else:
                with st.spinner("🔍 Enhancing prompt..."):
                    refined = enhance_prompt(raw_prompt)
                    if refined.startswith("ERROR:"):
                        st.error(f"❌ {refined}")
                    else:
                        st.session_state.refined_prompt = refined
                        st.success("✅ Prompt refined!")
                        try:
                            # ==============================================
                            # LOGGING: log_event — If moving, keep inside correct try/except and scope
                            # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                            # ==============================================
                            log_event(st.session_state.auth_user["id"], "image_refine_prompt", mode="create", prompt=raw_prompt, prompt_len=len(raw_prompt), meta={"before": raw_prompt, "after": refined})
                        except Exception:
                            pass
                        st.rerun()

    # Generate
    with col2:
        if st.button("🎨 Generate", key="generate_img_btn", use_container_width=True):
            # Daily image quota (10/day, successes only)
            try:
                used = count_success_images_today(st.session_state.auth_user["id"])
                if used >= 10:
                    st.warning("Daily image quota reached (10). Ask admin to reset or try tomorrow.")
                    st.stop()
            except Exception:
                pass

            # Determine mode/prompt for logging
            _mode_key = "create" if mode == "Create" else ("inspire" if mode == "Inspire" else "combine")
            _prompt_for_log = ""
            if mode == "Create":
                _prompt_for_log = (st.session_state.get("refined_prompt") or st.session_state.get("image_raw_prompt") or "")
            elif mode == "Inspire":
                _prompt_for_log = st.session_state.get("img_prompt_inspire","")
            else:
                _prompt_for_log = st.session_state.get("img_prompt_combine","")

            try:
                # ==============================================
                # LOGGING: log_event — If moving, keep inside correct try/except and scope
                # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                # ==============================================
                log_event(st.session_state.auth_user["id"], "image_generate_click", mode=_mode_key, prompt=_prompt_for_log, prompt_len=len(_prompt_for_log))
            except Exception:
                pass

            with st.spinner("🎨 Generating your image..."):
                try:
                    if mode == "Create":
                        prompt_to_send = (
                            st.session_state.get("refined_prompt", "").strip()
                            or st.session_state.get("image_raw_prompt", "").strip()
                        )
                        if not prompt_to_send:
                            raise Exception("No prompt available.")
                        img_bytes = generate_flux(prompt_to_send)

                    elif mode == "Inspire":
                        if not input_bytes:
                            raise Exception("Please upload an image first.")
                        if not st.session_state.get("img_prompt_inspire", "").strip():
                            raise Exception("Please enter a prompt.")
                        b64 = base64.b64encode(input_bytes).decode()
                        uri = f"data:{input_mime};base64,{b64}"
                        img_bytes = generate_kontext_max(
                            st.session_state["img_prompt_inspire"].strip(),
                            uri
                        )

                    else:  # Combine Images mode
                        # Build list of files for upload
                        files_for_upload = []
                        
                        # Add chained image first if available
                        if prefilled:
                            from io import BytesIO
                            chained_file = BytesIO(prefilled)
                            chained_file.name = "chained_image.png"
                            chained_file.type = "image/png"
                            files_for_upload.append(chained_file)
                        
                        # Add uploaded files
                        if multi_files:
                            remaining_slots = max(0, 4 - len(files_for_upload))
                            files_for_upload.extend(multi_files[:remaining_slots])
                        
                        # Validation
                        if not files_for_upload:
                            raise Exception("Please upload at least 1 image (or reuse the previous output).")
                        
                        if not st.session_state.get("img_prompt_combine", "").strip():
                            raise Exception("Please enter a prompt.")
                        
                        # Generate image using multi-image kontext BASE64 version
                        img_bytes = generate_multi_image_kontext_base64(
                            prompt=st.session_state["img_prompt_combine"].strip(),
                            image_files=files_for_upload,
                            aspect_ratio=st.session_state.get("combine_aspect", "match_input_image"),
                            model_slug="flux-kontext-apps/multi-image-list"
                        )

                    # Store the generated image in session state for persistent display
                    st.session_state.generated_image = img_bytes
                    st.session_state.generation_success = True
                    st.session_state.generation_error = None
                    # Save image to Supabase Storage + DB (public URL) and log success
                    try:
                        _model_used = "flux-schnell" if mode == "Create" else ("flux-kontext-max" if mode == "Inspire" else "flux-kontext-multi")
                        img_id, img_url = save_image_return_url(
                            st.session_state.auth_user["id"],
                            st.session_state.generated_image,
                            _mode_key,
                            _model_used,
                            _prompt_for_log
                        )
                        if img_url:
                            st.session_state["last_image_url"] = img_url
                        # ==============================================
                        # LOGGING: log_event — If moving, keep inside correct try/except and scope
                        # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                        # ==============================================
                        log_event(st.session_state.auth_user["id"], "image_generate_done", mode=_mode_key, status="success", image_id=img_id, prompt=_prompt_for_log, prompt_len=len(_prompt_for_log), meta={"used_prompt": ("refined" if (mode=="Create" and st.session_state.get("refined_prompt")) else "raw"), "file": f"{img_id}.png", "url": img_url})
                    except Exception as _e:
                        st.warning(f"Image saved locally but analytics/storage failed: {_e}")


                except Exception as e:
                    st.session_state.generation_success = False
                    st.session_state.generation_error = str(e)
                    st.session_state.generated_image = None

                    try:
                        # ==============================================
                        # LOGGING: log_event — If moving, keep inside correct try/except and scope
                        # (Non-executable guidance. Do not remove. Keep indentation consistent.)
                        # ==============================================
                        log_event(st.session_state.auth_user["id"], "image_generate_done", mode=_mode_key, status="error", prompt=_prompt_for_log, prompt_len=len(_prompt_for_log), meta={"error": str(e)})
                    except Exception:
                        pass
# Display results outside the spinner and button logic
    if st.session_state.get("generation_success"):
        if st.session_state.get("last_image_url"):
            st.link_button("Open in browser", st.session_state["last_image_url"])
        st.success("✅ Image generated successfully!")
        
        # Display the image persistently
        if st.session_state.get("generated_image"):
            st.image(
                st.session_state.generated_image, 
                caption="Generated Image",
                use_container_width=True
            )
            
            # Download button
            st.download_button(
                label="📥 Download Image",
                data=st.session_state.generated_image,
                file_name="generated_image.png",
                mime="image/png",
                key="download_image_btn",
                use_container_width=True
            )
            
            # Edit/Chain button (only for Inspire and Combine Images modes)
            current_mode = st.session_state.get("img_mode", mode)
            if current_mode in ("Inspire", "Combine Images"):
                if st.button("✏️ Edit This Image", key="edit_img_btn", use_container_width=True):
                    st.session_state.chained_image = st.session_state.generated_image
                    st.session_state.edit_mode = current_mode
                    st.success("✅ Image saved for editing! You can now change settings and generate again.")
                    st.rerun()

    elif st.session_state.get("generation_success") == False:
        # Show error if generation failed
        error_msg = st.session_state.get("generation_error", "Unknown error occurred")
        st.error(f"❌ {error_msg}")

    # Initialize session state variables
    if "generation_success" not in st.session_state:
        st.session_state.generation_success = None
    if "generated_image" not in st.session_state:
        st.session_state.generated_image = None

    if "generation_error" not in st.session_state:
        st.session_state.generation_error = None

# ---- Footer ----
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; font-size: 0.9rem;'>Powered by image generation AI models </div>",
    unsafe_allow_html=True
)
