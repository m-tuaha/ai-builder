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
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates
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
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import colorsys  # for RGB↔HSV
import hashlib
import random

#You can Disable streamlit file watching function by uncommenting below line:
#st.set_option("server.fileWatcherType", "none")

# ---- Font options for text overlay feature ----
FONT_OPTIONS = {
    "Arial": "fonts/arial.ttf",
    "Times New Roman": "fonts/times.ttf",
    "Verdana": "fonts/verdana.ttf",
    "Georgia": "fonts/georgia.ttf",
    "Courier New": "fonts/cour.ttf",
    "Comic Sans MS": "fonts/comic.ttf",
    "Reckless": "fonts/reckless.ttf",
}

# ---- Load system prompts from external markdown files ----
@st.cache_data
def load_prompt(file_path):
    """Load a text/markdown prompt file into a string."""
    return Path(file_path).read_text(encoding="utf-8")

TEXT_PROMPT_PATH = Path(__file__).parent / "prompts" / "text_system.md"
IMAGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "image_system.md"
SOCIAL_PROMPT_PATH = Path(__file__).parent / "prompts" / "social_system.md"
STYLE_CARD_PROMPT_SOCIAL_PATH = Path(__file__).parent / "prompts" / "style_social.md"


TEXT_SYSTEM_PROMPT = load_prompt(TEXT_PROMPT_PATH)
IMAGE_SYSTEM_PROMPT = load_prompt(IMAGE_PROMPT_PATH)
SOCIAL_SYSTEM_PROMPT = load_prompt(SOCIAL_PROMPT_PATH)
STYLE_CARD_CREATOR_PROMPT_SOCIAL = load_prompt(STYLE_CARD_PROMPT_SOCIAL_PATH)

# Initialize Image-Generator session state
if "img_mode" not in st.session_state:
    st.session_state.img_mode = "Create"
if "chained_image" not in st.session_state:
    st.session_state.chained_image = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = None
if "inspire_prompt" not in st.session_state:
    st.session_state.inspire_prompt = ""
    
# ===== INSPIRE templates loader =====
@st.cache_data
def load_inspire_templates():
    try:
        import json
        from pathlib import Path
        path = Path(__file__).parent / "prompts" / "inspire_templates.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Failed to load inspire templates: {e}")
    return []

# ----- preview template images before and after if path has image uploaded -----
# ----- preview template images before/after (robust resolver) -----
def _resolve_path_or_url(path_or_url: str | None) -> str | None:
    """
    Returns an HTTP/HTTPS/Data URL string usable in <img src="...">.
    - If given an http(s) or data: URL, returns as-is.
    - If given a local relative path next to this app, returns a base64 data URL.
    """
    if not path_or_url:
        return None
    s = str(path_or_url).strip()
    if s.startswith(("http://", "https://", "data:")):
        return s

    p = (Path(__file__).parent / s).resolve()
    if not p.exists():
        return None

    # Guess MIME type
    ext = p.suffix.lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".gif":
        mime = "image/gif"

    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _render_preview_pair(tmpl: dict, small_height: int = 110):
    """Render tiny Before→After strip if preview images exist."""
    b = _resolve_path_or_url(tmpl.get("preview_before"))
    a = _resolve_path_or_url(tmpl.get("preview_after"))
    if not (b and a):
        return  # nothing to show

    c1, c2, c3 = st.columns([1, 0.2, 1])
    with c1:
        st.caption("Before")
        st.image(b, use_container_width=True, output_format="auto", clamp=True)
    with c2:
        st.markdown("<div style='text-align:center; font-weight:700; margin-top:34px;'>→</div>", unsafe_allow_html=True)
    with c3:
        st.caption("After")
        st.image(a, use_container_width=True, output_format="auto", clamp=True)
        
def _render_preview_pair_fixed(tmpl: dict):
    """
    Render fixed-size before/after thumbs (no captions, no modal).
    Supports either:
      - flat keys: preview_before / preview_after
      - nested keys: preview: { before, after }
    """
    # Try flat keys first (your JSON uses these)
    b = _resolve_path_or_url(tmpl.get("preview_before"))
    a = _resolve_path_or_url(tmpl.get("preview_after"))

    # Fallback to nested structure
    if not (b and a):
        prev = tmpl.get("preview") or {}
        b = b or _resolve_path_or_url(prev.get("before"))
        a = a or _resolve_path_or_url(prev.get("after"))
        
def _parse_palette(raw: str) -> list[str]:
    if not raw:
        return []

    allowed_names = {
        "WHITE",
        "BLACK",
        "RED",
        "BLUE",
        "GREEN",
        "YELLOW",
        "ORANGE",
        "PURPLE",
        "PINK",
        "BROWN",
        "GRAY",
        "GREY",
        "TEAL",
        "NAVY",
        "GOLD",
        "SILVER",
        "CREAM",
        "IVORY",
        "BEIGE",
        "CYAN",
        "MAGENTA",
    }

    tokens: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        upper = token.upper()
        if re.fullmatch(r"#([0-9A-F]{3}|[0-9A-F]{6})", upper):
            if len(upper) == 4:
                upper = "#" + "".join(ch * 2 for ch in upper[1:])
            tokens.append(upper)
        elif upper in allowed_names:
            tokens.append(upper)
        if len(tokens) >= 3:
            break

    if len(tokens) == 1:
        single = tokens[0]
        extras: list[str] = []
        if single.startswith("#") and len(single) == 7:
            r = int(single[1:3], 16)
            g = int(single[3:5], 16)
            b = int(single[5:7], 16)
            factor = 0.7
            darker = f"#{max(0, int(r * factor)):02X}{max(0, int(g * factor)):02X}{max(0, int(b * factor)):02X}"
            if darker not in tokens:
                extras.append(darker)
        elif single not in extras:
            extras.append(single)
        extras.append("WHITE")
        tokens.extend([c for c in extras if c not in tokens])
        tokens = tokens[:3]

    return tokens


def _palette_snippet(colors: list[str]) -> str:
    if colors:
        return f"( {', '.join(colors)} )"
    return "in complementary tones to the product"
    
    if not (b and a):
        return  # nothing to show

    st.markdown(
        f"""
        <div class="tmpl-card">
          <div class="tmpl-row">
            <div class="tmpl-col">
              <span class="tmpl-label">Before</span>
              <img class="tmpl-thumb" src="{b}" alt="before">
            </div>
            <div class="tmpl-arrow">→</div>
            <div class="tmpl-col">
              <span class="tmpl-label">After</span>
              <img class="tmpl-thumb" src="{a}" alt="after">
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

def generate_flux(
    prompt: str,
    aspect_ratio: str | None = None,
    seed: int | None = None,
    output_format: str = "png",
    prompt_upsampling: bool | None = None,
) -> bytes:
    """Call Replicate Flux Schnell API and return image bytes."""
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {"input": {"prompt": prompt}}
    if aspect_ratio and aspect_ratio != "match_input_image":
        payload["input"]["aspect_ratio"] = aspect_ratio
    if seed is not None and str(seed).strip() != "":
        # Replicate expects integer; will cast safely
        try:
            payload["input"]["seed"] = int(str(seed).strip())
        except ValueError:
            pass
    if output_format in {"png", "jpg"}:
        payload["input"]["output_format"] = output_format
    if prompt_upsampling is True:
        payload["input"]["prompt_upsampling"] = True
    
    api_endpoint = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"
    
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
        
# --- ADD: Gemini via Replicate helper (Create mode only) ---
def generate_gemini_replicate(
    prompt: str,
    output_format: str = "png",
) -> bytes:
    """
    Call Replicate google/gemini-2.5-flash-image and return image bytes.
    Only supports prompt + output_format; AR/seed/upsampling are ignored by this model on Replicate.
    """
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {"input": {"prompt": prompt}}
    ofmt = (output_format or "png").lower()
    if ofmt in {"png", "jpg"}:
        payload["input"]["output_format"] = ofmt

    api_endpoint = "https://api.replicate.com/v1/models/google/gemini-2.5-flash-image/predictions"

    try:
        # Kick off prediction
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()
        prediction_id = prediction["id"]

        # Poll until completion
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                image_url = outputs[0] if isinstance(outputs, list) else outputs
                img_resp = requests.get(image_url)
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
        
        
def generate_kontext_max(
    prompt: str,
    input_image_uri: str,
    aspect_ratio: str | None = None,
    seed: str | int | None = None,
    output_format: str | None = None,
    prompt_upsampling: bool | None = None,
) -> bytes:

    """Call Replicate Flux Kontext Max API and return image bytes."""
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"input": {"prompt": prompt, "input_image": input_image_uri}}
    # AR: keep match_input_image as "no explicit AR"
    if aspect_ratio and aspect_ratio != "match_input_image":
        payload["input"]["aspect_ratio"] = aspect_ratio
    # seed
    if seed is not None and str(seed).strip() != "":
        try:
            payload["input"]["seed"] = int(str(seed).strip())
        except ValueError:
            pass
    # output format
    ofmt = (output_format or "png").lower()
    if ofmt in {"png", "jpg"}:
        payload["input"]["output_format"] = ofmt
    # prompt upsampling
    if prompt_upsampling is True:
        payload["input"]["prompt_upsampling"] = True
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
        
        
def generate_nano_banana_replicate(
    prompt: str,
    image_data_urls: list[str] | None = None,
    aspect_ratio: str = "match_input_image",    
    output_format: str = "png",
) -> bytes:
    """
    Call Replicate google/nano-banana and return image bytes.
    Accepts 0+ image URIs (data: URLs or https URLs).
    Supports aspect ratio and output format fields; seed/prompt_upsampling remain unsupported.
    """
    token = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    ofmt = (output_format or "png").lower()
    if ofmt not in {"png", "jpg"}:
        ofmt = "png"

    payload_input = {
        "prompt": prompt,
        "output_format": ofmt,
        "aspect_ratio": aspect_ratio or "match_input_image",
    }    
    
    if image_data_urls:
        payload_input["image_input"] = image_data_urls

    payload = {"input": payload_input}
    api_endpoint = "https://api.replicate.com/v1/models/google/nano-banana/predictions"

    try:
        # Kick off
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()
        prediction_id = prediction["id"]

        # Poll
        max_wait_time = 300
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}", headers=headers
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                image_url = outputs[0] if isinstance(outputs, list) else outputs
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return img_resp.content

            if status_data["status"] == "failed":
                raise Exception(f"Prediction failed: {status_data.get('error', 'Unknown error')}")

            time.sleep(2)

        raise Exception("Generation timed out after 5 minutes")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Replicate API request error: {e}")
    except Exception as e:
        raise Exception(f"Image generation error: {e}")

def generate_multi_image_kontext_base64(
    prompt: str,
    image_files,
    aspect_ratio: str = "match_input_image",
    model_slug: str = "flux-kontext-apps/multi-image-kontext-max",
    seed: int | None = None,
    output_format: str = "png",
) -> bytes:
    """
    Calls Replicate 'flux-kontext-apps/multi-image-kontext-max' using base64 data URLs.
    Requires exactly 2 input images (the model schema's minimum).
    Returns image bytes.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is required.")

    if not image_files or len(image_files) < 2:
        raise ValueError("Please provide at least 2 images for Combine Images.")

    token = st.secrets["REPLICATE_API_TOKEN"]

    try:
        # Convert exactly first two images to base64 data URLs
        image_data_urls = []
        for f in image_files[:2]:
            try:
                if hasattr(f, "seek"):
                    f.seek(0)
            except Exception:
                pass

            # Read file data
            if hasattr(f, "read"):
                file_data = f.read()
                try:
                    if hasattr(f, "seek"):
                        f.seek(0)
                except Exception:
                    pass
                content_type = getattr(f, "type", None)
            else:
                # bytes already
                file_data = f
                content_type = None

            # Fallback MIME if missing/odd
            if not content_type:
                content_type = "image/png"
            elif content_type == "image/jpg":
                content_type = "image/jpeg"

            b64_data = base64.b64encode(file_data).decode("utf-8")
            data_url = f"data:{content_type};base64,{b64_data}"
            image_data_urls.append(data_url)

        if len(image_data_urls) < 2:
            raise ValueError("At least 2 images are required for this model.")

        # Build payload per model schema (separate fields)
        payload_input = {
            "prompt": prompt.strip(),
            "input_image_1": image_data_urls[0],
            "input_image_2": image_data_urls[1],
            "aspect_ratio": aspect_ratio,                # e.g. 'match_input_image', '1:1', '16:9', ...
            "output_format": (output_format or "png").lower(),  # 'png' or 'jpg'
            "safety_tolerance": 2,
        }

        # Optional seed
        if seed is not None and str(seed).strip() != "":
            try:
                payload_input["seed"] = int(str(seed).strip())
            except Exception:
                pass

        payload = {"input": payload_input}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }

        api_endpoint = f"https://api.replicate.com/v1/models/{model_slug}/predictions"
        resp = requests.post(api_endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        prediction = resp.json()

        # Poll
        prediction_id = prediction["id"]
        max_wait_time = 300
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_resp = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "succeeded":
                outputs = status_data["output"]
                image_url = outputs[0] if isinstance(outputs, list) else outputs
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                return img_resp.content

            if status_data["status"] == "failed":
                raise Exception(f"Prediction failed: {status_data.get('error', 'Unknown error')}")

            time.sleep(2)

        raise Exception("Generation timed out after 5 minutes")

    except Exception as e:
        raise Exception(f"Multi-image generation error: {str(e)}")


#---- PIL helper: draw text overlay ----
def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    
#---- Text Overlay Helper Function  ----
def add_text_overlay(
    img_bytes: bytes,
    text: str,
    font_size: int,
    color_hex: str,
    pos_x_pct: int,
    pos_y_pct: int,
    font_path: str | None = None,
    outline: bool = True,
    outline_width: int = 2,
    outline_color: str = "#000000",
) -> bytes:
    """Draws text on image at percentage position. Returns PNG bytes."""
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = base.size
    x = int(W * (pos_x_pct / 100.0))
    y = int(H * (pos_y_pct / 100.0))

    # Try chosen font; fall back gracefully
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)  # common on Linux
    except Exception:
        font = ImageFont.load_default()

    # Prep colors
    def _hex_to_rgba(hex_color: str, a: int = 255):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, a)

    fill_rgba = _hex_to_rgba(color_hex, 255)
    outline_rgba = _hex_to_rgba(outline_color, 220)

    # Draw on separate layer then composite (keeps transparency)
    txt_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(txt_layer)

    if outline and outline_width > 0:
        d.text(
            (x, y),
            text,
            font=font,
            fill=fill_rgba,
            anchor="mm",
            stroke_width=int(outline_width),
            stroke_fill=outline_rgba,
        )
    else:
        d.text((x, y), text, font=font, fill=fill_rgba, anchor="mm")

    out = Image.alpha_composite(base, txt_layer).convert("RGBA")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
    
    
# Crop/Resize Helper Functions
def _parse_preset_size(preset: str) -> tuple[int, int] | None:
    """Known pixel presets; return (w,h) or None if custom/aspect-only."""
    presets = {
        "Instagram Square (1080x1080)": (1080, 1080),
        "YouTube Thumbnail (1280x720)": (1280, 720),
        "LinkedIn Banner (1584x396)": (1584, 396),
        "Story/Reel (1080x1920)": (1080, 1920),
        "Twitter/X Header (1500x500)": (1500, 500),
        "WhatsApp Template (1125x600)": (1125, 600),
        "Viber Template (1600x1200)": (1600, 1200),
    }
    return presets.get(preset)

def _parse_aspect_ratio(s: str) -> tuple[int, int] | None:
    """'1:1' -> (1,1); '16:9' -> (16,9); 'match_input' -> None."""
    if s in ("match_input_image", "None", "", None):
        return None
    try:
        a, b = s.split(":")
        return (max(1, int(a)), max(1, int(b)))
    except Exception:
        return None

def _center_crop_to_aspect(img: Image.Image, aspect: tuple[int, int]) -> Image.Image:
    """Crop (center) to reach desired aspect, keeping max content."""
    w, h = img.size
    ar_w, ar_h = aspect
    target_ratio = ar_w / ar_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # too wide -> crop sides
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # too tall -> crop top/bottom
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

def _anchor_crop_to_aspect(img: Image.Image, aspect: tuple[int,int], ax_pct: int, ay_pct: int) -> Image.Image:
    """Crop to aspect with adjustable anchor (ax_pct, ay_pct in 0..100)."""
    w, h = img.size
    ar_w, ar_h = aspect
    target_ratio = ar_w / ar_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # crop width
        new_w = int(h * target_ratio)
        max_left = w - new_w
        left = int((ax_pct/100.0) * max_left)
        return img.crop((left, 0, left + new_w, h))
    else:
        # crop height
        new_h = int(w / target_ratio)
        max_top = h - new_h
        top = int((ay_pct/100.0) * max_top)
        return img.crop((0, top, w, top + new_h))

def resize_crop_image(
    img_bytes: bytes,
    method: str,                    # "Contain (no crop)" or "Cover (crop to fill)"
    preset: str,                    # one of dropdown or "Custom"
    out_w: int | None,
    out_h: int | None,
    aspect_choice: str,             # "match_input_image" or "1:1" etc.
    anchor_x_pct: int = 50,         # only for Cover when aspect provided
    anchor_y_pct: int = 50,
    resample: int = Image.LANCZOS,
) -> bytes:
    """
    Returns PNG bytes resized/cropped.
    method:
      - Contain (no crop): fit entire image inside target; pads/can letterbox if aspect differs.
      - Cover (crop to fill): crop to target aspect then resize to exact output.
    """
    import io
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = base.size

    # 1) Decide target size
    target = _parse_preset_size(preset)
    if target:
        tw, th = target
    else:
        # Custom numeric, fallback gracefully
        tw = max(1, int(out_w)) if out_w else W
        th = max(1, int(out_h)) if out_h else H

    # 2) Aspect handling
    aspect = _parse_aspect_ratio(aspect_choice)
    img2 = base

    if method.startswith("Cover"):
        # crop to aspect first (if given), else infer from output tw x th
        asp = aspect or (tw, th)
        img2 = _anchor_crop_to_aspect(base, asp, anchor_x_pct, anchor_y_pct)
        img2 = img2.resize((tw, th), resample=resample)

    else:  # Contain (no crop)
        # if aspect given and differs, first center-crop to aspect? No: 'Contain' promises no crop.
        # We just thumbnail to fit and then paste onto transparent canvas if needed.
        img2 = base.copy()
        img2.thumbnail((tw, th), resample=resample)
        canvas = Image.new("RGBA", (tw, th), (0,0,0,0))
        off_x = (tw - img2.width) // 2
        off_y = (th - img2.height) // 2
        canvas.paste(img2, (off_x, off_y))
        img2 = canvas

    # 3) Encode PNG
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return buf.getvalue()


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

# ---- Custom CSS for inspire templates
st.markdown("""
    <style>
      .tmpl-card { margin: 6px 0 14px 0; }
      .tmpl-row { display: flex; align-items: center; gap: 14px; }
      .tmpl-col { display: flex; flex-direction: column; gap: 6px; }
      .tmpl-label { color: #6b7280; font-size: 0.85rem; }
      /* No-crop thumbnail: keeps uniform size, adds subtle letterbox if needed */
      .tmpl-thumb {
        width: 240px;
        aspect-ratio: 3 / 2;     /* 480x320 */
        height: auto;            /* computed from aspect-ratio */
        border-radius: 12px;
        object-fit: contain;     /* <-- key change: no cropping */
        background: #f1f5f9;     /* soft slate to show letterbox bars */
        display: block;
        box-shadow: 0 2px 10px rgba(0,0,0,0.10);
      }
      .tmpl-arrow { font-size: 20px; color: #6b7280; padding: 0 6px; }
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


# ---- Initialize session state (no chat_history here) ----
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

# Product Hero state
st.session_state.setdefault("ph_image_bytes", None)
st.session_state.setdefault("ph_image_mime", None)
st.session_state.setdefault("ph_palette_input", "")
st.session_state.setdefault("ph_palette_list", [])
st.session_state.setdefault("ph_engine", "Model B")
st.session_state.setdefault("ph_ar", "match_input_image")
st.session_state.setdefault("ph_outfmt", "jpg")
st.session_state.setdefault("ph_gallery", [])
st.session_state.setdefault("ph_selected_idx", None)
st.session_state.setdefault("ph_edit_choice", None)
st.session_state.setdefault("ph_edit_manual", "")
st.session_state.setdefault("ph_clear_manual", False)
st.session_state.setdefault("ph_pending_mode", None)

# Migrate legacy variant storage to the new gallery structure
if "ph_variants" in st.session_state:
    legacy_variants = st.session_state.get("ph_variants") or []
    if legacy_variants and not st.session_state.get("ph_gallery"):
        converted = []
        for item in legacy_variants:
            if not isinstance(item, dict):
                continue
            converted.append(
                {
                    "img": item.get("img"),
                    "format": item.get("format", "png"),
                    "engine": item.get("engine", "B"),
                    "seed": item.get("seed"),
                }
            )
        st.session_state.ph_gallery = converted
    del st.session_state["ph_variants"]
    
    
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

PH_LOGS_DISABLED = bool(st.secrets.get("DISABLE_PRODUCT_HERO_LOGS", True))

def ph_log(user_id, action, **props):
    if PH_LOGS_DISABLED:
        return
    try:
        log_event(user_id, action, **props)
    except Exception:
        pass

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
    
    
def get_global_image_quota() -> int:
    """Global per-user daily image cap. Configurable via Streamlit secret GLOBAL_IMAGE_QUOTA."""
    try:
        val = int(st.secrets.get("GLOBAL_IMAGE_QUOTA", 10))
        return max(1, val)
    except Exception:
        return 10
    
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
          .in_("action", ["text_generate_done","image_generate_done","image_refine_prompt","text_refine_prompt"])
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
        quota = get_global_image_quota()
        st.write(f"Images today: **{used} / {quota}**")
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
                _t = meta.get("text")
                if isinstance(_t, list):
                    # Show up to 2 variants, then (+N more)
                    shown = [_short(x or "") for x in _t[:2]]
                    extra = len(_t) - 2
                    details = "\n---\n".join(shown) + (f"\n(+{extra} more)" if extra > 0 else "")
                else:
                    details = _short(_t or "")

            elif result == "Image":
                details = meta.get("file") or meta.get("url") or ""

            elif result == "Refine":
                # For both image_refine_prompt and text_refine_prompt
                after = meta.get("after", "")
                details = _short(after)

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
            _full = _m.get("error") or _m.get("after") or _m.get("url") or _m.get("text") or ""
            if isinstance(_full, list):
                _full = "\n\n---\n\n".join(_full)
            st.text_area("Full Details", _full, height=180)
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

tab1, tab2, tab3 = st.tabs(["📝 Text Generator", "🎨 Image Generator", "🛍️ Product Hero"])

# ==============================================
# ---- CONTENT GENERATOR TAB ----
# ==============================================
# TEXT TAB START — OTT + Social (routing + schemas)
# ==============================================
with tab1:
    
    # --- Campaign Type selector (outside the form so it reruns immediately) ---
    if "campaign_type" not in st.session_state:
        st.session_state.campaign_type = "OTT / Direct Messaging"

    st.session_state.campaign_type = st.radio(
        "Campaign Type",
        ["OTT / Direct Messaging", "Social Media Post"],
        horizontal=True,
        key="campaign_type_radio",
)
    
    # ---- Input Form ----
    with st.form("campaign_form"):
        st.subheader("Campaign Details")

        # Read the selection the user made outside the form
        campaign_type = st.session_state.get("campaign_type", "OTT / Direct Messaging")

        if campaign_type == "OTT / Direct Messaging":
            # --- OTT fields ---
            channel = st.selectbox("Channel", ["whatsapp", "sms", "viber"], key="ott_channel")
            prompt = st.text_area(
                "Campaign Instruction / Prompt",
                placeholder="Describe your campaign, product details, offer, and any special instructions.",
                key="ott_prompt",
            )
            language = st.text_input("Language", "en", key="ott_language")
            tone = st.text_input("Tone", "friendly", key="ott_tone")
            max_length = st.number_input("Max Length", min_value=1, max_value=1024, value=600, key="ott_max_len")
            variants = st.number_input("Number of Variants", min_value=1, max_value=3, value=1, key="ott_variants")
            platform = None  # not used for OTT

        else:
            # --- Social fields ---
            platform = st.selectbox("Platform", ["linkedin", "facebook", "viva_engage"], key="social_platform")
            prompt = st.text_area(
                "Campaign Instruction / Prompt",
                placeholder="Describe the announcement, target audience, key points, and any special instructions.",
                key="social_prompt",
            )
            language = st.text_input("Language", "en", key="social_language")
            tone = st.text_input("Tone", "professional", key="social_tone")
            max_length = st.number_input("Max Length", min_value=100, max_value=3000, value=1000, key="social_max_len")
            variants = 1  # fixed for social
            channel = "social_post"  # internal routing key
            
            # --- Copy My Style (Social) ---
            with st.expander("Copy My Style (Social) — optional"):
                # defaults
                st.session_state.setdefault("social_samples_text", "")
                st.session_state.setdefault("style_card_social_text", "")
                st.session_state.setdefault("use_style_card_social", False)

                # One textarea for up to 4 samples (delimiter optional); HARD CAP 5000 chars
                samples_text = st.text_area(
                    "Paste 1–4 sample posts (separate with a line containing only ---). Max 5000 characters.",
                    key="social_samples_text",
                    height=160,
                    max_chars=5000,   # UI hard cap
                    help="Tip: Keep brand names/links if needed; the builder generalizes patterns."
                )
                # Extra belt-and-suspenders: clamp server-side to 5000 as well
                if samples_text and len(samples_text) > 5000:
                    st.session_state.social_samples_text = samples_text[:5000]

                colA, colB = st.columns([1,1])
                with colA:
                    build_card_btn = st.form_submit_button("Build Style Card")
                with colB:
                    # Enable the toggle only if a card exists; also handle post-build auto-enable
                    has_card = bool(st.session_state.get("style_card_social_text"))
                    if st.session_state.pop("pending_enable_style_card_social", False):
                        st.session_state["use_style_card_social"] = True

                    st.checkbox(
                        "Use this style for Social now",
                        key="use_style_card_social",
                        disabled=not has_card
                    )

                # If a card already exists, show a read-only preview + hash
                if st.session_state.get("style_card_social_text"):
                    _card = st.session_state["style_card_social_text"]
                    st.text_area("Current Style Card (preview)", value=_card, height=120, disabled=True)
                    st.caption(f"Length: {len(_card)} chars")

                # Handle build action
                if build_card_btn:
                    raw = (st.session_state.get("social_samples_text") or "").strip()
                    if not raw:
                        st.warning("Please paste at least one sample.")
                        

                    # Prepare user message for builder
                    user_blob = f"SAMPLES START\n{raw[:5000]}\nSAMPLES END"

                    try:
                        openai_api_key = st.secrets["OPENAI_API_KEY"]
                        client_sc = openai.OpenAI(api_key=openai_api_key)

                        resp_sc = client_sc.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": STYLE_CARD_CREATOR_PROMPT_SOCIAL},
                                {"role": "user", "content": user_blob}
                            ],
                            max_tokens=600,
                            temperature=0.2,
                        )
                        card = (resp_sc.choices[0].message.content or "").strip().replace("\n", " ")
                        if len(card) > 1000:
                            card = card[:1000].rstrip()

                        if not card or card.upper() == "INSUFFICIENT_SAMPLES":
                            st.warning("Could not build a style card from the provided samples.")

                        # Save to session + compute stable token/hash
                        st.session_state.style_card_social_text = card
                        _hash = hashlib.sha1(card.encode("utf-8")).hexdigest()[:12]
                        st.session_state.style_card_social_hash = _hash

                        # Log creation (no schema change)
                        try:
                            log_event(
                                st.session_state.auth_user["id"],
                                "text_stylecard_created",
                                mode="text",
                                model="gpt-4o-mini",
                                prompt="style_card_social",
                                prompt_len=len(raw),
                                meta={
                                    "samples_count": 1 + raw.count("\n---\n"),
                                    "channel": "social",
                                    "style_card_hash": _hash,
                                    "chars": len(card),
                                },
                            )
                        except Exception:
                            pass

                        # Enable toggle on next run and rerun NOW
                        st.session_state["pending_enable_style_card_social"] = True
                        st.rerun()
                        
                    except Exception:
                        # Fail gracefully—don’t block text generation if style-card build trips
                        st.warning("Style card build failed; continuing without it.")
                        pass
            # Style status (Social) near Generate
            _has_card = bool(st.session_state.get("style_card_social_text"))
            _style_on = _has_card and st.session_state.get("use_style_card_social", False)
            st.caption("Style: ON (Social)" if _style_on else "Style: OFF")

        generate_btn = st.form_submit_button("Generate Content")

    # ---- GENERATE CONTENT: starts a NEW session ----
    if generate_btn and prompt:
        # Log text generate click (does not count toward quota)
        try:
            log_event(
                st.session_state.auth_user["id"],
                "text_generate_click",
                mode="text",
                prompt=prompt,
                prompt_len=len(prompt),
            )
        except Exception:
            pass

        openai_api_key = st.secrets["OPENAI_API_KEY"]
        client = openai.OpenAI(api_key=openai_api_key)

        # ---- Choose system prompt (fallback to TEXT if SOCIAL not loaded) ----
        sys_prompt_for_social = globals().get("SOCIAL_SYSTEM_PROMPT")
        system_prompt = sys_prompt_for_social if (channel == "social_post" and sys_prompt_for_social) else TEXT_SYSTEM_PROMPT

        # ---- Start fresh conversation for this generation ----
        st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

        # ---- Build user payload per type ----
        if channel == "social_post":
            input_json = {
                "prompt": prompt,
                "channel": "social_post",
                "platform": platform,          # 'linkedin' | 'facebook' | 'viva_engage'
                "language": language,
                "tone": tone,
                "maxLength": int(max_length),
                "variants": 1
            }
            
            # Optionally include Style Card for Social initial generate
            _card = st.session_state.get("style_card_social_text")
            if _card and len(_card) > 1000:
                _card = _card[:1000].rstrip()
            if _card and st.session_state.get("use_style_card_social", False):
                input_json["style_card"] = _card
                
            
        else:
            input_json = {
                "prompt": prompt,
                "channel": channel,            # 'whatsapp' | 'sms' | 'viber'
                "language": language,
                "tone": tone,
                "maxLength": int(max_length),
                "variants": int(variants)
            }

        # ---- Append the user message (JSON) ----
        try:
            st.session_state.chat_history.append(
                {"role": "user", "content": safe_json_dumps(input_json)}
            )
        except Exception as e:
            st.error(f"Error preparing request: {e}")
            st.stop()

        try:
            # ---- OpenAI call ----
            num = 1 if channel == "social_post" else int(variants)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.chat_history,
                max_tokens=2000,
                temperature=0.7,
                n=num
            )

            # ---- Collect variants (Social = 1) ----
            variant_list = []
            for i in range(num):
                output = response.choices[i].message.content.strip()

                # Debug: Show raw output
                with st.expander(f"Debug: Raw GPT Output for Variant {i+1}"):
                    st.text(output)

                try:
                    # Try direct JSON first
                    if output.startswith('['):
                        arr = json.loads(output)
                        result = arr[i] if i < len(arr) else arr[0] if arr else create_fallback_response()
                    else:
                        result = json.loads(output)

                    # Validate/fix (will add placeholders: [] if missing — OK for Social)
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

            # ---- Reset chat_history to system + user + assistant (selected variant) ----
            st.session_state.chat_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_json_dumps(input_json)},
                {"role": "assistant", "content": safe_json_dumps(st.session_state.last_output)}
            ]

            # ---- Store RAW INPUT/OUTPUT for debug ----
            st.session_state.raw_input_text = safe_json_dumps(st.session_state.chat_history)
            st.session_state.raw_output_text = safe_json_dumps(st.session_state.last_output)

            st.success("Content generated successfully!")

            # ---- Log success (Admin Details shows meta.text) ----
            try:
                # Build compact request JSON for Admin "Prompt" column
                compact_request = {
                    "channel": input_json.get("channel"),
                    "platform": input_json.get("platform"),
                    "language": input_json.get("language"),
                    "tone": input_json.get("tone"),
                    "maxLength": input_json.get("maxLength"),
                    "variants": input_json.get("variants"),
                    "prompt_head": (prompt[:120] + "…") if len(prompt) > 120 else prompt,
                    "prompt_len": len(prompt),
                    "style_card": bool(input_json.get("style_card")),  # <-- added boolean
                }

                # Gather all variant bodies for Details (can be 1 or many)
                all_bodies = [v.get("body", "") for v in st.session_state.last_variants]

                log_event(
                    st.session_state.auth_user["id"],
                    "text_generate_done",
                    mode="text",
                    status="success",
                    prompt=safe_json_dumps(compact_request),
                    prompt_len=len(prompt),
                    meta={"text": all_bodies}  # list of strings
)
            except Exception:
                pass

        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
            try:
                # Also store a compact request JSON even on error (helps debugging)
                compact_request = {
                    "channel": input_json.get("channel"),
                    "platform": input_json.get("platform"),
                    "language": input_json.get("language"),
                    "tone": input_json.get("tone"),
                    "maxLength": input_json.get("maxLength"),
                    "variants": input_json.get("variants"),
                    "prompt_head": (prompt[:120] + "…") if len(prompt) > 120 else prompt,
                    "prompt_len": len(prompt),
                    "style_card": bool(input_json.get("style_card")),
                }
                log_event(
                    st.session_state.auth_user["id"],
                    "text_generate_done",
                    mode="text",
                    status="error",
                    prompt=safe_json_dumps(compact_request),
                    prompt_len=len(prompt),
                    meta={"error": str(e)}
                )
            except Exception:
                pass

    # ---- Variant selector if multiple (OTT only) ----
    if st.session_state.last_variants:
        if len(st.session_state.last_variants) > 1:
            options = [f"Variant {i+1}" for i in range(len(st.session_state.last_variants))]
            selected = st.selectbox("Select Variant to View/Edit", options,
                                    index=st.session_state.selected_variant)
            idx = options.index(selected)
            st.session_state.last_output = st.session_state.last_variants[idx]
            st.session_state.selected_variant = idx

            # Update chat_history assistant slot with newly selected variant
            if "chat_history" in st.session_state and st.session_state.chat_history:
                try:
                    _sys = st.session_state.chat_history[0]["content"]
                    _user = st.session_state.chat_history[1]["content"]
                except Exception:
                    _sys = TEXT_SYSTEM_PROMPT
                    _user = safe_json_dumps({})
                st.session_state.chat_history = [
                    {"role": "system", "content": _sys},
                    {"role": "user", "content": _user},
                    {"role": "assistant", "content": safe_json_dumps(st.session_state.last_output)},
                ]
                # update debug stash
                st.session_state.raw_input_text = safe_json_dumps(st.session_state.chat_history)

    # ---- OUTPUT section: Body (+ Placeholders if present for OTT) ----
    if st.session_state.last_output:
        output = st.session_state.last_output
        st.markdown("### Generated Content")

        # Unescape the body content for display
        display_body = unescape_json_string(output.get("body", ""))
        
        # --- STATE MIRROR: keep widget state in sync with selected/edited variant ---
        # Streamlit ignores `value=` when a widget key already exists. We update session_state
        # *before* rendering widgets so the textarea reflects the newly selected/edited variant.
        if st.session_state.get("body_out") != display_body:
            st.session_state["body_out"] = display_body
 
        _length_str = str(output.get("length", ""))
        if st.session_state.get("length_out") != _length_str:
            st.session_state["length_out"] = _length_str
 
        _variant_id_str = output.get("variant_id", "")
        if st.session_state.get("variant_id_out") != _variant_id_str:
            st.session_state["variant_id_out"] = _variant_id_str

        # Now render widgets bound to the mirrored state
        body = st.text_area("Body", st.session_state.get("body_out", ""), height=120, key="body_out")
        length = st.text_input("Length", st.session_state.get("length_out", ""), key="length_out", disabled=True)
        variant_id = st.text_input("Variant ID", st.session_state.get("variant_id_out", ""), key="variant_id_out", disabled=True)
        
        
        placeholders = output.get("placeholders", [])
        if placeholders:  # Social will naturally show nothing because it's []
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
                # Always use the currently selected variant (last_output), not a stale index-2 message
                previous_output_content = safe_json_dumps(st.session_state.last_output)

                followup_message = {
                    "role": "user",
                    "content": safe_json_dumps({
                        "edit_instruction": follow_up,
                        "base_campaign": json.loads(base_user_content),
                        "previous_output": json.loads(previous_output_content)
                    })
                }
                st.session_state.chat_history.append(followup_message)

                # ---- OpenAI edit call ----
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

                # Collapse chat history to minimal triple with the latest result
                try:
                    _sys = st.session_state.chat_history[0]["content"]
                    _user = st.session_state.chat_history[1]["content"]
                except Exception:
                    _sys = TEXT_SYSTEM_PROMPT
                    _user = safe_json_dumps({})

                st.session_state.chat_history = [
                    {"role": "system", "content": _sys},
                    {"role": "user", "content": _user},
                    {"role": "assistant", "content": safe_json_dumps(result)},
                ]

                st.session_state.last_output = result
                if st.session_state.last_variants:
                    idx = st.session_state.selected_variant
                    st.session_state.last_variants[idx] = result

                # Debug stash
                st.session_state.raw_input_text = safe_json_dumps(st.session_state.chat_history)
                st.session_state.raw_output_text = safe_json_dumps(result)
                
                # ---- LOG: text_refine_prompt (success) ----
                try:
                    # Build a compact "prompt" object for the Admin table (no long bodies)
                    _base = json.loads(base_user_content)
                    _prev = json.loads(previous_output_content)  # has variant_id
                    compact_edit_prompt = {
                        "action": "edit",
                        "edit_instruction": follow_up[:240],
                        "channel": _base.get("channel"),
                        "platform": _base.get("platform"),
                        "language": _base.get("language"),
                        "tone": _base.get("tone"),
                        "maxLength": _base.get("maxLength"),
                        "base_variant_id": _prev.get("variant_id"),
                    }
                    log_event(
                        st.session_state.auth_user["id"],
                        "text_refine_prompt",
                        mode="text",
                        status="success",
                        prompt=safe_json_dumps(compact_edit_prompt),
                        meta={
                            "before": json.loads(previous_output_content).get("body", ""),
                            "after": result.get("body", "")
                        }
                    )
                except Exception:
                    pass
                st.rerun()

            except Exception as e:
                # ------------------------------
                # 1b) LOG: text_refine_prompt (error)
                # ------------------------------
                try:
                    # base_user_content / previous_output_content may exist if failure happened late;
                    # guard with try/except to avoid masking the original error.
                    try:
                        _base = json.loads(base_user_content)
                    except Exception:
                        _base = {}
                    try:
                        _prev = json.loads(previous_output_content)
                    except Exception:
                        _prev = {}

                    compact_edit_prompt = {
                        "action": "edit",
                        "edit_instruction": follow_up[:240],
                        "channel": _base.get("channel"),
                        "platform": _base.get("platform"),
                        "language": _base.get("language"),
                        "tone": _base.get("tone"),
                        "maxLength": _base.get("maxLength"),
                        "base_variant_id": _prev.get("variant_id"),
                    }
                    log_event(
                        st.session_state.auth_user["id"],
                        "text_refine_prompt",
                        mode="text",
                        status="error",
                        prompt=safe_json_dumps(compact_edit_prompt),
                        meta={"error": str(e)}
                    )
                except Exception:
                    pass

                # UI error messages (keep these last)
                st.error(f"Edit Error: {e}")
                st.error(f"Error details: {str(e)}")

# ---- IMAGE GENERATOR TAB ----
# ==============================================
# IMAGE TAB START — Keep Replicate/Flux + upload; add UI only
# (Non-executable guidance. Do not remove. Keep indentation consistent.)
# ==============================================
with tab2:
    st.subheader("Image Generation Details")

    # Modes: Create (text->image), Inspire (style copy from single template), Combine Images (multi-image model)
    pending_mode = st.session_state.pop("ph_pending_mode", None)
    if pending_mode:
        st.session_state["img_mode"] = pending_mode
        st.session_state.edit_mode = pending_mode    
    mode = st.selectbox("Mode", ["Create", "Inspire", "Combine Images", "Quick Actions"], key="img_mode")

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
            
        # --- Engine selection for Create mode ---
        st.session_state.setdefault("create_engine", "Model A")
        st.selectbox(
            "Engine",
            ["Model A", "Model B"],
            key="create_engine",
            help="Choose which image model to use for Create."
        )
   
        # --- Advanced options (Create) ---
        with st.expander("Advanced options"):
            st.session_state.setdefault("create_aspect", "1:1")
            st.session_state.setdefault("create_seed", "")
            st.session_state.setdefault("create_outfmt", "png")
            st.session_state.setdefault("create_prompt_upsampling", False)

            st.selectbox(
                "Aspect ratio",
                ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9", "9:21", "2:1", "1:2"],
                index=0,
                key="create_aspect",
            )
            st.text_input("Seed (optional)", key="create_seed", placeholder="e.g., 1234")
            #Disabled the output image option
            #st.selectbox("Output type", ["png", "jpg"], index=0, key="create_outfmt") 
            st.checkbox("Creative boost (let AI enhance the prompt)", value=False, key="create_prompt_upsampling")
            st.caption("If unsupported by the selected engine, this setting is ignored.")
            
            # Clarify unsupported params for Gemini on Replicate
            if st.session_state.get("create_engine") == "Model B":
                st.caption("Note: Aspect ratio, Seed, and Creative boost are ignored by Model B")            

    # ---------- INSPIRE (template-assisted or freeform) ----------
    elif mode == "Inspire":
        # input image
        if st.session_state.get("chained_image") and st.session_state.get("edit_mode") == mode:
            input_bytes = st.session_state.chained_image
            input_mime = "image/png"
            st.image(input_bytes, caption="Using previous output", use_container_width=True)
        else:
            uploaded = st.file_uploader(
                "Upload an image to edit / to copy style from",
                type=["png", "jpg", "webp"],
                key="input_image_file_inspire",
            )
            if uploaded:
                input_bytes = uploaded.read()
                input_mime = uploaded.type
                st.image(input_bytes, caption="Uploaded image", use_container_width=True)
            else:
                input_bytes, input_mime = None, None
                
        # --- Engine selection for Inspire mode ---
        st.session_state.setdefault("inspire_engine", "Model A")
        st.selectbox(
            "Engine",
            ["Model A", "Model B"],
            key="inspire_engine",
            help="Choose which image model to use for Inspire (single-image edit)."
        )        

        st.markdown("### Prompt Templates (optional)")
        templates = load_inspire_templates()
        if templates:
            cols = st.columns(2)
            for i, tmpl in enumerate(templates):
                c = cols[i % 2]
                with c:
                    st.markdown(f"**{tmpl['title']}**")
                    st.caption(tmpl.get("desc", ""))

                    # Fixed-size before/after thumbnails (always uniform height)
                    try:
                        _render_preview_pair_fixed(tmpl)
                    except Exception:
                        pass

                    if st.button("Use Example", key=f"use_{tmpl['id']}"):
                        ex = (tmpl.get("example") or "").strip()
                        if ex:
                            st.session_state.inspire_prompt = ex        

        # Make the text area key unique to this mode (and user) to avoid collisions
        ta_key = f"inspire_text_area__{st.session_state.get('img_mode','Inspire')}__{st.session_state.get('auth_user',{}).get('id','')}"
        final_prompt = st.text_area(
            "Prompt for Kontext",
            value=st.session_state.inspire_prompt,
            placeholder="Type your own prompt here, or use a template above",
            height=120,
            key=ta_key
        )
        st.session_state.inspire_prompt = final_prompt
        # --- Advanced options (Inspire) ---
        with st.expander("Advanced options"):
            st.session_state.setdefault("inspire_aspect", "match_input_image")
            st.session_state.setdefault("inspire_seed", "")
            st.session_state.setdefault("inspire_outfmt", "png")
            st.session_state.setdefault("inspire_prompt_upsampling", False)

            st.selectbox(
                "Aspect ratio",
                ["match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9", "9:21", "2:1", "1:2"],
                index=0,
                key="inspire_aspect",
            )
            st.text_input("Seed (optional)", key="inspire_seed", placeholder="e.g., 1234")
            #Disabled the output image option from advanced options
            st.selectbox("Output type", ["png", "jpg"], index=0, key="inspire_outfmt")
            st.checkbox("Creative boost (let AI enhance the prompt)", value=False, key="inspire_prompt_upsampling")
            #st.caption("If unsupported by the selected engine, this setting is ignored.")

            # Clarify unsupported params for Nano-Banana on Replicate
            if st.session_state.get("inspire_engine") == "Model B":
                st.caption("Note: Aspect ratio, Seed, and Creative boost are ignored by Model B.")            
        


    # ---------- COMBINE IMAGES (multi‑image kontext) ----------
    elif mode == "Combine Images":
        st.caption("Upload up to 2 images. The model will combine/transform them per your prompt.")

        # If chaining from previous output, show it and allow up to 3 more uploads
        prefilled = st.session_state.get("chained_image") if st.session_state.get("edit_mode") == "Combine Images" else None
        if prefilled:
            st.image(prefilled, caption="Using previous output (counts as 1 image)", use_container_width=True)

        multi_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "webp"],
            accept_multiple_files=True,
            key="input_images_combine",
        )

        max_total_images = 2
        allowed_uploads = max_total_images - (1 if prefilled else 0)
        if allowed_uploads < 0:
            allowed_uploads = 0

        if multi_files and len(multi_files) > allowed_uploads:
            st.warning(
                f"Only {max_total_images} images are supported in Combine mode. "
                f"Using the first {allowed_uploads} upload(s); please remove extras if needed."
            )
            multi_files = list(multi_files)[:allowed_uploads]

        st.session_state.setdefault("combine_engine", "Model A")
        st.selectbox(
            "Engine",
            ["Model A", "Model B"],
            key="combine_engine",
            help="Choose the model used for combining images. Model A - flux, Model B - Google",
        )
        
        prompt_combine = st.text_input(
            "Enter your prompt",
            key="img_prompt_combine",
            placeholder="e.g., Put the product from image 1 on the background of image 2 with a summer vibe",
        )

        # --- Advanced options (Combine) ---
        with st.expander("Advanced options"):
            st.session_state.setdefault("combine_aspect", "match_input_image")
            st.session_state.setdefault("combine_seed", "")
            st.session_state.setdefault("combine_outfmt", "png")
            st.session_state.setdefault("combine_prompt_upsampling", False)

            st.selectbox(
                "Aspect ratio",
                ["match_input_image", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9", "9:21", "2:1", "1:2"],
                index=0,
                key="combine_aspect",
            )
            st.text_input("Seed (optional)", key="combine_seed", placeholder="e.g., 1234")
            #Disabled the output image option from advanced options
            #st.selectbox("Output type", ["png", "jpg"], index=0, key="combine_outfmt")
            st.checkbox("Creative boost (let AI enhance the prompt)", value=False, key="combine_prompt_upsampling")
            st.caption("If unsupported by the selected engine, this setting is ignored.")
            
            if st.session_state.get("combine_engine") == "Model B":
                st.caption("Note: Aspect ratio, Seed, and Creative boost are ignored by Model B.")            
        
    # ---------- QUICK ACTIONS (Pillow) ----------
    elif mode == "Quick Actions":
        qa = st.selectbox("Choose an action", ["Add Text Overlay", "Resize/Crop"], key="qa_choice")
        
        # ----- Source image: last generated or upload -----
        src_choice = st.radio("Source image", ["Use last generated image", "Upload new image"], horizontal=True)
        src_bytes = None
        src_mime = "image/png"

        if src_choice == "Use last generated image":
            if st.session_state.get("generated_image"):
                src_bytes = st.session_state.generated_image
                st.image(src_bytes, caption="Using last generated image", use_container_width=True)
            else:
                st.info("No previous image found. Please upload one.")
        else:
            up = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"], key="qa_upload")
            if up:
                src_bytes = up.read()
                src_mime = up.type
                st.image(src_bytes, caption="Uploaded image", use_container_width=True)

        st.divider()

        if qa == "Add Text Overlay":
            # Font family selection
            font_label = st.selectbox("Font", list(FONT_OPTIONS.keys()), index=0, key="qa_font_family")
            font_path = FONT_OPTIONS.get(font_label)

            # Outline options
            add_outline   = st.checkbox("Add outline", value=True, key="qa_outline")
            outline_width = st.slider("Outline width (px)", 0, 12, 2, step=1, key="qa_outline_width")
            outline_color = st.color_picker("Outline color", value="#000000", key="qa_outline_color")

            colA, colB = st.columns([2, 1])
            with colA:
                overlay_text = st.text_input("Text to overlay", placeholder="Type your headline or caption…", key="qa_text")
                color = st.color_picker("Text color", value="#FFFFFF", key="qa_color")
            with colB:
                font_size = st.slider("Font size", 12, 160, 48, step=2, key="qa_fontsize")

            st.caption("Position (percentage of image width/height)")
            cx = st.slider("X position %", 0, 100, 50, key="qa_x")
            cy = st.slider("Y position %", 0, 100, 50, key="qa_y")

            # ============= LIVE PREVIEW (uses the helper directly) =============
            if src_bytes:
                if overlay_text and overlay_text.strip():
                    try:
                        preview_bytes = add_text_overlay(
                            img_bytes=src_bytes,
                            text=overlay_text.strip(),
                            font_size=int(font_size),
                            color_hex=color,
                            pos_x_pct=int(cx),
                            pos_y_pct=int(cy),
                            font_path=font_path,
                            outline=bool(add_outline),
                            outline_width=int(outline_width),
                            outline_color=outline_color,
                        )
                        st.markdown("**Live Preview:** (updates as you type/slide)")
                        st.image(preview_bytes, use_container_width=True)
                    except Exception as _pe:
                        st.warning(f"Preview failed: {_pe}")
                else:
                    st.info("Enter text to see a live preview here.")
            else:
                st.info("Select or upload an image to enable live preview.")
            # =======================================================

            apply = st.button("✅ Apply Text Overlay", use_container_width=True, key="apply_text_overlay")

            if apply:
                if not src_bytes:
                    st.error("Please select or upload an image first.")
                    st.stop()
                if not overlay_text or not overlay_text.strip():
                    st.error("Please enter text.")
                    st.stop()

                # Quota check (respects exclusion list)
                try:
                    used = count_success_images_today(st.session_state.auth_user["id"])
                    quota = get_global_image_quota()
                    if used >= quota:
                        st.warning(f"Daily image quota reached ({quota}). Ask admin to reset or try tomorrow.")
                        st.stop()
                except Exception:
                    pass

                try:
                    with st.spinner("Rendering overlay…"):
                        out_bytes = add_text_overlay(
                            img_bytes=src_bytes,
                            text=overlay_text.strip(),
                            font_size=int(font_size),
                            color_hex=color,
                            pos_x_pct=int(cx),
                            pos_y_pct=int(cy),
                            font_path=font_path,
                            outline=bool(add_outline),
                            outline_width=int(outline_width),
                            outline_color=outline_color,
                        )

                    # persist in session so users can chain / preview
                    st.session_state.generated_image = out_bytes
                    st.session_state.generation_success = True
                    st.session_state.generation_error = None

                    # Save + log as a normal image success (mode=text_overlay, model=pillow)
                    uid = st.session_state.auth_user["id"]
                    prompt_for_log = f'Text overlay: "{overlay_text.strip()}"'
                    try:
                        img_id, img_url = save_image_return_url(
                            uid, out_bytes, mode="text_overlay", model="pillow", prompt=prompt_for_log
                        )
                        if img_url:
                            st.session_state["last_image_url"] = img_url

                        log_event(
                            uid,
                            "image_generate_done",
                            mode="text_overlay",
                            status="success",
                            image_id=img_id,
                            prompt=prompt_for_log,
                            prompt_len=len(prompt_for_log),
                            meta={
                                "file": f"{img_id}.png",
                                "url": img_url,
                                "text": overlay_text.strip(),
                                "font": font_label,
                                "outline": bool(add_outline),
                                "outline_width": int(outline_width),
                                "outline_color": outline_color,
                            },
                        )
                    except Exception as _e:
                        st.warning(f"Image saved locally but analytics/storage failed: {_e}")

                except Exception as e:
                    st.session_state.generation_success = False
                    st.session_state.generation_error = str(e)
                    try:
                        log_event(
                            st.session_state.auth_user["id"],
                            "image_generate_done",
                            mode="text_overlay",
                            status="error",
                            prompt="Text overlay failed",
                            prompt_len=0,
                            meta={"error": str(e)},
                        )
                    except Exception:
                        pass
                    st.error(f"❌ {e}")
                    
                                                
        #Crop/Resize mode when selected
        if qa == "Resize/Crop":
            # Presets & method
            preset = st.selectbox(
                "Preset size",
                [
                    "Instagram Square (1080x1080)",
                    "Story/Reel (1080x1920)",
                    "YouTube Thumbnail (1280x720)",
                    "LinkedIn Banner (1584x396)",
                    "Twitter/X Header (1500x500)",
                    "WhatsApp Template (1125x600)",
                    "Viber Template (1600x1200)",
                    "Custom"
                ],
                index=0,
                key="rc_preset"
            )

            # ---------- NON-CUSTOM: keep your existing Cover/Contain flow ----------
            if preset != "Custom":
                method = st.radio(
                    "Method",
                    ["Contain (no crop)", "Cover (crop to fill)"],
                    horizontal=True,
                    key="rc_method"
                )

                # Aspect choice (optional)
                aspect = st.selectbox(
                    "Aspect ratio (optional)",
                    ["match_input_image", "1:1", "16:9", "9:16", "4:5", "5:4", "3:2", "2:3"],
                    index=0,
                    key="rc_aspect"
                )

                # Anchor sliders (only meaningful for Cover)
                ax, ay = 50, 50
                if method.startswith("Cover"):
                    st.caption("Crop anchor (only for Cover)")
                    ax = st.slider("Anchor X % (left→right)", 0, 100, 50, key="rc_ax")
                    ay = st.slider("Anchor Y % (top→bottom)", 0, 100, 50, key="rc_ay")

                # ===== Live Preview =====
                if src_bytes:
                    try:
                        preview_bytes = resize_crop_image(
                            img_bytes=src_bytes,
                            method=method,
                            preset=preset,
                            out_w=None, out_h=None,
                            aspect_choice=aspect,
                            anchor_x_pct=ax, anchor_y_pct=ay,
                        )
                        st.markdown("**Live Preview:**")
                        st.image(preview_bytes, use_container_width=True)
                    except Exception as _pe:
                        st.warning(f"Preview failed: {_pe}")
                else:
                    st.info("Select or upload an image to enable live preview.")

                apply_rc = st.button("✅ Apply Resize/Crop", use_container_width=True, key="apply_resize_crop")
                if apply_rc:
                    if not src_bytes:
                        st.error("Please select or upload an image first.")
                        st.stop()

                    # Quota check (respects exclusion list)
                    try:
                        used = count_success_images_today(st.session_state.auth_user["id"])
                        quota = get_global_image_quota()
                        if used >= quota:
                            st.warning(f"Daily image quota reached ({quota}). Ask admin to reset or try tomorrow.")
                            st.stop()
                    except Exception:
                        pass

                    try:
                        with st.spinner("Processing…"):
                            out_bytes = resize_crop_image(
                                img_bytes=src_bytes,
                                method=method,
                                preset=preset,
                                out_w=None, out_h=None,
                                aspect_choice=aspect,
                                anchor_x_pct=ax, anchor_y_pct=ay,
                            )

                        # Persist for global renderer
                        st.session_state.generated_image = out_bytes
                        st.session_state.generation_success = True
                        st.session_state.generation_error = None

                        # Save + log
                        uid = st.session_state.auth_user["id"]
                        prompt_for_log = f"Resize/Crop via {method}"
                        img_id, img_url = save_image_return_url(
                            uid, out_bytes, mode="resize_crop", model="pillow", prompt=prompt_for_log
                        )
                        if img_url:
                            st.session_state["last_image_url"] = img_url

                        log_event(
                            uid,
                            "image_generate_done",
                            mode="resize_crop",
                            status="success",
                            image_id=img_id,
                            prompt=prompt_for_log,
                            prompt_len=len(prompt_for_log),
                            meta={
                                "file": f"{img_id}.png",
                                "url": img_url,
                                "method": method,
                                "preset": preset,
                                "aspect": aspect,
                                "anchor_x_pct": ax,
                                "anchor_y_pct": ay,
                            },
                        )

                        st.success("Done.")

                    except Exception as e:
                        st.session_state.generation_success = False
                        st.session_state.generation_error = str(e)
                        try:
                            log_event(
                                st.session_state.auth_user["id"],
                                "image_generate_done",
                                mode="resize_crop",
                                status="error",
                                prompt="Resize/Crop failed",
                                prompt_len=0,
                                meta={"error": str(e)},
                            )
                        except Exception:
                            pass
                        st.error(f"❌ {e}")
                        

            # ---------- CUSTOM: use streamlit-cropper (no X/Y anchors) ----------
            else:
                # Safe local imports with aliases to avoid name shadowing
                try:
                    from streamlit_cropper import st_cropper as _st_cropper
                    import io as _io
                    from PIL import Image as PILImage
                except Exception as _e:
                    st.error("streamlit-cropper/Pillow not available. Add 'streamlit-cropper' to requirements.txt and redeploy.")
                    st.stop()

                st.caption("Drag the handles to choose your crop (freeform).")

                try:
                    pil_src = PILImage.open(_io.BytesIO(src_bytes)).convert("RGB")

                    # return_type must be 'image', 'box', or 'both'
                    cropped = _st_cropper(
                        pil_src,
                        realtime_update=True,
                        box_color="#00483b",
                        aspect_ratio=None,     # freeform crop
                        return_type="image",
                        key="rc_cropper"
                    )

                    st.markdown("**Live Preview:**")
                    st.image(cropped, use_container_width=True)

                    # Optional export size (defaults to cropped size)
                    colw, colh = st.columns(2)
                    with colw:
                        export_w = st.number_input("Export width (px)", min_value=64, max_value=4096,
                                                   value=int(cropped.size[0]), key="rc_exp_w")
                    with colh:
                        export_h = st.number_input("Export height (px)", min_value=64, max_value=4096,
                                                   value=int(cropped.size[1]), key="rc_exp_h")

                    apply_rc_custom = st.button("✅ Apply Crop", use_container_width=True, key="apply_resize_crop_custom")
                    if apply_rc_custom:
                        # Quota check
                        try:
                            used = count_success_images_today(st.session_state.auth_user["id"])
                            quota = get_global_image_quota()
                            if used >= quota:
                                st.warning(f"Daily image quota reached ({quota}). Ask admin to reset or try tomorrow.")
                                st.stop()
                        except Exception:
                            pass

                        try:
                            out_img = cropped
                            if (int(export_w), int(export_h)) != cropped.size:
                                out_img = cropped.resize((int(export_w), int(export_h)), PILImage.LANCZOS)

                            buf = _io.BytesIO()
                            out_img.save(buf, format="PNG")
                            out_bytes = buf.getvalue()

                            st.session_state.generated_image = out_bytes
                            st.session_state.generation_success = True
                            st.session_state.generation_error = None

                            uid = st.session_state.auth_user["id"]
                            prompt_for_log = "Resize/Crop via Custom (streamlit-cropper)"
                            img_id, img_url = save_image_return_url(
                                uid, out_bytes, mode="resize_crop", model="pillow", prompt=prompt_for_log
                            )
                            if img_url:
                                st.session_state["last_image_url"] = img_url

                            log_event(
                                uid, "image_generate_done", mode="resize_crop", status="success",
                                image_id=img_id, prompt=prompt_for_log, prompt_len=len(prompt_for_log),
                                meta={"file": f"{img_id}.png", "url": img_url, "preset": "Custom",
                                      "export_w": int(export_w), "export_h": int(export_h)}
                            )

                            st.success("Done.")

                        except Exception as e:
                            st.session_state.generation_success = False
                            st.session_state.generation_error = str(e)
                            try:
                                log_event(
                                    st.session_state.auth_user["id"], "image_generate_done",
                                    mode="resize_crop", status="error",
                                    prompt="Resize/Crop (Custom) failed", prompt_len=0,
                                    meta={"error": str(e)}
                                )
                            except Exception:
                                pass
                            st.error(f"❌ {e}")

                except Exception as _ce:
                    st.error(f"Cropper failed to load: {_ce}")

    # ---------- Reset & Refine / Generate buttons ----------
    col1, col2 = st.columns([1, 1])

    # Reset All
    with col1:
        if st.button("🔄 Reset All", key="reset_all", use_container_width=True):
            
            for k in [
                "image_raw_prompt", "refined_prompt", "image_editable_prompt",
                "chained_image", "edit_mode",
                "inspire_prompt", "img_prompt_combine", "img_mode", "combine_aspect",
                "create_engine",    # (already added in Create step)
                "inspire_engine"    # <-- add this
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
        if mode != "Quick Actions" and st.button("🎨 Generate", key="generate_img_btn", use_container_width=True):
            # Daily image quota (10/day, successes only)
            try:
                used = count_success_images_today(st.session_state.auth_user["id"])
                quota = get_global_image_quota()
                if used >= quota:
                    st.warning(f"Daily image quota reached ({quota}). Ask admin to reset or try tomorrow.")
                    st.stop()
            except Exception:
                pass

            # Determine mode/prompt for logging
            _mode_key = "create" if mode == "Create" else ("inspire" if mode == "Inspire" else "combine")
            _prompt_for_log = ""
            if mode == "Create":
                _prompt_for_log = (st.session_state.get("refined_prompt") or st.session_state.get("image_raw_prompt") or "")
            elif mode == "Inspire":
                _prompt_for_log = st.session_state.get("inspire_prompt","")
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

                        _engine = st.session_state.get("create_engine", "Model A")
                        if "Model B" in _engine:
                            # Gemini on Replicate ignores AR/seed/boost; force PNG for storage consistency
                            img_bytes = generate_gemini_replicate(
                                prompt_to_send,
                                output_format="png"
                            )
                        else:
                            # Flux path (existing behavior)
                            img_bytes = generate_flux(
                                prompt_to_send,
                                aspect_ratio=st.session_state.get("create_aspect"),
                                seed=st.session_state.get("create_seed"),
                                output_format=st.session_state.get("create_outfmt"),
                                prompt_upsampling=st.session_state.get("create_prompt_upsampling"),
                            )                    

                    elif mode == "Inspire":
                        if not input_bytes:
                            raise Exception("Please upload an image first.")
                        final = st.session_state.get("inspire_prompt","").strip()
                        if not final:
                            raise Exception("Please enter a prompt.")

                        # Build data: URL for the single input image
                        b64 = base64.b64encode(input_bytes).decode()
                        uri = f"data:{input_mime};base64,{b64}"

                        _ieng = st.session_state.get("inspire_engine", "Model A")
                        if "Model B" in _ieng:
                            # Force PNG for consistency; Nano-Banana ignores AR/Seed/Boost
                            img_bytes = generate_nano_banana_replicate(
                                prompt=final,
                                image_data_urls=[uri],
                                aspect_ratio=st.session_state.get("inspire_aspect", "match_input_image"),                                
                                output_format="png"
                            )
                        else:
                            # Existing Flux Kontext Max path
                            img_bytes = generate_kontext_max(
                                final, uri,
                                aspect_ratio=st.session_state.get("inspire_aspect"),
                                seed=st.session_state.get("inspire_seed"),
                                output_format=st.session_state.get("inspire_outfmt"),
                                prompt_upsampling=st.session_state.get("inspire_prompt_upsampling"),
                            )



                    elif mode == "Inspire":
                        if not input_bytes:
                            raise Exception("Please upload an image first.")
                        final = st.session_state.get("inspire_prompt","").strip()
                        if not final:
                            raise Exception("Please enter a prompt.")
                        b64 = base64.b64encode(input_bytes).decode()
                        uri = f"data:{input_mime};base64,{b64}"
                        img_bytes = generate_kontext_max(
                            final, uri,
                            aspect_ratio=st.session_state.get("inspire_aspect"),
                            seed=st.session_state.get("inspire_seed"),
                            output_format=st.session_state.get("inspire_outfmt"),
                            prompt_upsampling=st.session_state.get("inspire_prompt_upsampling"),
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
                            remaining_slots = max(0, 2 - len(files_for_upload))
                            files_for_upload.extend(list(multi_files)[:remaining_slots])
                        
                        # Validation
                        if len(files_for_upload) < 2:
                            raise Exception("Please provide at least 2 images (or include the chained image + 1 upload).")

                        if len(files_for_upload) > 2:
                            raise Exception("Please use no more than 2 images in Combine mode.")                            
                        
                        if not st.session_state.get("img_prompt_combine", "").strip():
                            raise Exception("Please enter a prompt.")
                        
                        engine_choice = st.session_state.get("combine_engine", "Model A")

                        if "Model B" in engine_choice:
                            data_urls = []
                            for f in files_for_upload:
                                try:
                                    if hasattr(f, "seek"):
                                        f.seek(0)
                                except Exception:
                                    pass

                                if hasattr(f, "read"):
                                    file_data = f.read()
                                    try:
                                        if hasattr(f, "seek"):
                                            f.seek(0)
                                    except Exception:
                                        pass
                                    mime_type = getattr(f, "type", None)
                                else:
                                    file_data = f
                                    mime_type = None

                                if not mime_type:
                                    mime_type = "image/png"
                                elif mime_type == "image/jpg":
                                    mime_type = "image/jpeg"

                                b64 = base64.b64encode(file_data).decode("utf-8")
                                data_urls.append(f"data:{mime_type};base64,{b64}")

                            if len(data_urls) != 2:
                                raise Exception("Model B requires exactly 2 images.")

                            img_bytes = generate_nano_banana_replicate(
                                prompt=st.session_state["img_prompt_combine"].strip(),
                                image_data_urls=data_urls,
                                aspect_ratio=st.session_state.get("combine_aspect", "match_input_image"),                                
                                output_format=st.session_state.get("combine_outfmt", "png"),
                            )
                        else:
                            # Generate image using multi-image kontext BASE64 version
                            img_bytes = generate_multi_image_kontext_base64(
                                prompt=st.session_state["img_prompt_combine"].strip(),
                                image_files=files_for_upload,
                                aspect_ratio=st.session_state.get("combine_aspect", "match_input_image"),
                                model_slug="flux-kontext-apps/multi-image-kontext-max",
                                seed=st.session_state.get("combine_seed"),
                                output_format=st.session_state.get("combine_outfmt", "png"),
                            )                        
                    # Store the generated image in session state for persistent display
                    st.session_state.generated_image = img_bytes
                    st.session_state.generation_success = True
                    st.session_state.generation_error = None
                    # Save image to Supabase Storage + DB (public URL) and log success
                    try:                      
                        
                        if mode == "Create":
                            _model_used = (
                                "gemini-2.5-flash-image"
                                if st.session_state.get("create_engine","Model A") == "Model B"
                                else "flux-1.1-pro"
                            )
                        elif mode == "Inspire":
                            _model_used = (
                                "nano-banana"
                                if st.session_state.get("inspire_engine","Model A") == "Model B"
                                else "flux-kontext-max"
                            )
                        elif mode == "Combine Images":
                            _model_used = (
                                "nano-banana"
                                if st.session_state.get("combine_engine","Model A") == "Model B"
                                else "flux-kontext-multi"
                            )
                        else:
                            _model_used = "flux-kontext-multi"

                        if not _model_used:
                            _model_used = "flux-1.1-pro" if mode == "Create" else (
                                "flux-kontext-max" if mode == "Inspire" else "flux-kontext-multi"
                            )
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
                        adv = {}
                        if _mode_key == "create":
                            _engine = st.session_state.get("create_engine","Model A")
                            adv = {
                                "engine": ("gemini" if "Model B" in _engine else "flux"),
                                "aspect": st.session_state.get("create_aspect"),
                                "seed": st.session_state.get("create_seed"),
                                "output_type": ("png" if "Model B" in _engine else st.session_state.get("create_outfmt")),
                                "prompt_upsampling": bool(st.session_state.get("create_prompt_upsampling")),
                            }
                            if "Model B" in _engine:
                                adv["ignored_params"] = ["aspect","seed","prompt_upsampling"]
                        elif _mode_key == "inspire":
                            _ieng = st.session_state.get("inspire_engine","Model A")
                            adv = {
                                "engine": ("nano-banana" if "Model B" in _ieng else "flux"),
                                "aspect": st.session_state.get("inspire_aspect"),
                                "seed": st.session_state.get("inspire_seed"),
                                "output_type": (
                                    "png" if "Model B" in _ieng else st.session_state.get("inspire_outfmt")
                                ),
                 
                                "prompt_upsampling": bool(st.session_state.get("inspire_prompt_upsampling")),
                            }
                            if "Model B" in _ieng:
                                adv["ignored_params"] = ["aspect","seed","prompt_upsampling"]
                        elif _mode_key == "combine":
                            _ceng = st.session_state.get("combine_engine","Model A")
                            adv = {
                                "engine": ("nano-banana" if "Model B" in _ceng else "flux"),
                                "aspect": st.session_state.get("combine_aspect"),
                                "seed": st.session_state.get("combine_seed"),
                                "output_type": st.session_state.get("combine_outfmt"),
                                "prompt_upsampling": bool(st.session_state.get("combine_prompt_upsampling")),
                            }
                            if "Model B" in _ceng:
                                adv["ignored_params"] = ["aspect","seed","prompt_upsampling"]                                
                        
                        
                        meta_payload = {
                            "used_prompt": ("refined" if (mode=="Create" and st.session_state.get("refined_prompt")) else "raw"),
                            "file": f"{img_id}.png",
                            "url": img_url,
                            **adv,
                        }

                        log_event(
                            st.session_state.auth_user["id"], "image_generate_done",
                            mode=_mode_key, status="success", image_id=img_id, prompt=_prompt_for_log,
                            prompt_len=len(_prompt_for_log), meta=meta_payload
                        )
                        
                        # Clear prompts after a successful Create generation so the next run uses fresh input
                        if mode == "Create":
                            for k in ("refined_prompt", "image_raw_prompt", "image_editable_prompt"):
                                st.session_state.pop(k, None)
                                
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

with tab3:
    uid = st.session_state.get("auth_user", {}).get("id")
    st.markdown(
        """
        <style>
        div[data-testid="stTabContent"][aria-label="🛍️ Product Hero"] > div > div {
            max-width: 1100px;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stTabContent"][aria-label="🛍️ Product Hero"] div[data-testid="column"] {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    existing_gallery = st.session_state.get("ph_gallery", [])
    if (
        st.session_state.get("ph_selected_idx") is not None
        and (st.session_state.ph_selected_idx >= len(existing_gallery))
    ):
        st.session_state.ph_selected_idx = None

    left_col, center_col, right_col = st.columns([1.1, 1.2, 0.9])

    with left_col:
        st.markdown("### Upload Product Image")
        uploaded_product = st.file_uploader(
            "Upload Product Image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="ph_upload_file",
        )
        if uploaded_product is not None:
            product_bytes = uploaded_product.read()
            if len(product_bytes) > 10 * 1024 * 1024:
                st.error("File exceeds 10MB limit. Please choose a smaller file.")
            else:
                st.session_state.ph_image_bytes = product_bytes
                mime = uploaded_product.type or "image/png"
                if mime == "image/jpg":
                    mime = "image/jpeg"
                st.session_state.ph_image_mime = mime

        if st.session_state.ph_image_bytes:
            st.image(
                st.session_state.ph_image_bytes,
                caption="Product Image",
                use_container_width=True,
            )

        st.text_input(
            "Brand Colors (optional)",
            key="ph_palette_input",
            placeholder="#00c072, #00483b, white",
        )
        st.caption(
            "Optional: add up to 3 brand colors (e.g., `#00c072, #00483b, white`). We’ll blend them into a smooth studio gradient."
        )
        st.session_state.ph_palette_list = _parse_palette(
            st.session_state.get("ph_palette_input", "")
        )
        palette_colors = st.session_state.get("ph_palette_list", [])
        if palette_colors:
            sw_cols = st.columns(len(palette_colors))
            for col, color in zip(sw_cols, palette_colors):
                text_color = "#000000"
                if color.startswith("#") and len(color) == 7:
                    try:
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
                        text_color = "#000000" if luminance > 0.6 else "#FFFFFF"
                    except ValueError:
                        text_color = "#000000"
                with col:

                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid rgba(0,0,0,0.15);
                            border-radius:6px;
                            padding:6px;
                            text-align:center;
                            font-size:0.75rem;
                            background:{color};
                            color:{text_color};
                        ">
                            {color}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )                

        st.radio(
            "Generation Engine",
            options=["Model A", "Model B"],
            format_func=lambda opt: (
                "Model A (Flux)"
                if opt == "Model A"
                else "Model B (Gemini)"
            ),
            key="ph_engine",
        )
        if st.session_state.ph_engine == "Model B":
            st.selectbox(
                "Aspect Ratio",
                ["match_input_image", "1:1", "4:5", "9:16", "16:9"],
                key="ph_ar",
            )
            st.radio(
                "Output",
                options=["jpg", "png"],
                key="ph_outfmt",
                horizontal=True,
            )

        if st.button("Generate Hero", key="ph_generate", use_container_width=True):
            if not st.session_state.ph_image_bytes:
                st.warning("Please upload a product image first.")
            else:
                palette_phrase = _palette_snippet(
                    st.session_state.get("ph_palette_list", [])
                )
                base_prompt = (
                    """Create a clean, scroll-stopping product hero using the uploaded image as the exact reference.
Center the product, remove clutter, preserve geometry and label fidelity.
Place on a subtle studio surface with a soft realistic shadow and gentle rim light.
Background: smooth gradient in brand-friendly colors %s with high contrast and no on-image text.
Photographic, crisp details, commercial polish, square 1:1, social-media ready."""
                    % palette_phrase
                )
                mime = st.session_state.get("ph_image_mime") or "image/png"
                if mime == "image/jpg":
                    mime = "image/jpeg"
                data_url = "data:%s;base64,%s" % (
                    mime,
                    base64.b64encode(st.session_state.ph_image_bytes).decode("utf-8"),
                )
                try:                    
                    engine_choice = st.session_state.get("ph_engine", "Model B")
                    with st.spinner("Generating hero..."):
                        if engine_choice == "Model A":
                            seed_val = random.randint(100000, 999999)
                            variant_bytes = generate_kontext_max(
                                base_prompt,
                                data_url,
                                aspect_ratio="1:1",
                                seed=seed_val,
                                output_format="png",
                            )
                            new_variant = {
                                "img": variant_bytes,
                                "seed": seed_val,
                                "engine": "A",
                                "format": "png",
                            }
                        else:
                            variant_bytes = generate_nano_banana_replicate(
                                prompt=base_prompt,
                                image_data_urls=[data_url],
                                aspect_ratio=st.session_state.get(
                                    "ph_ar", "match_input_image"
                                ),
                                output_format=st.session_state.get(
                                    "ph_outfmt", "jpg"
                                ),
                            )
                            new_variant = {
                                "img": variant_bytes,
                                "seed": None,
                                "engine": "B",
                                "format": st.session_state.get("ph_outfmt", "jpg"),
                            }
                    st.session_state.ph_gallery = [new_variant]
                    st.session_state.ph_selected_idx = 0
                    st.success("Hero ready!")
                    ph_log(
                        uid,
                        "image_generate_done",
                        mode="product_hero",
                        status="success",
                        model="A" if engine_choice == "Model A" else "B",
                        variant_count=1,
                    )
                except Exception as e:
                    st.error(f"❌ {e}")

    with center_col:
        st.markdown("### Gallery")
        ph_gallery = st.session_state.get("ph_gallery", [])
        selected_idx = st.session_state.get("ph_selected_idx")
        if not ph_gallery:
            st.caption("Your generated images will appear here.")
        else:
            num_cols = 2
            for row_start in range(0, len(ph_gallery), num_cols):
                row_items = list(enumerate(ph_gallery))[row_start : row_start + num_cols]
                cols = st.columns(num_cols)
                for col, (idx, variant) in zip(cols, row_items):
                    with col:
                        is_selected = selected_idx == idx
                        border_color = "#00c072" if is_selected else "rgba(0,0,0,0.1)"
                        st.markdown(
                            f"<div style='border:3px solid {border_color}; border-radius:12px; padding:10px;'>",
                            unsafe_allow_html=True,
                        )
                        st.image(variant.get("img"), use_container_width=True)
                        if variant.get("seed") is not None:
                            st.caption(f"Seed: {variant['seed']}")
                        button_label = "Selected" if is_selected else "Select"
                        if st.button(
                            button_label,
                            key=f"ph_select_{idx}",
                            disabled=is_selected,
                            use_container_width=True,
                        ):
                            st.session_state.ph_selected_idx = idx
                            st.rerun()
                        if is_selected:
                            st.caption("Currently selected")
                        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("### Actions")
        ph_gallery = st.session_state.get("ph_gallery", [])
        selected_idx = st.session_state.get("ph_selected_idx")
        if selected_idx is None or selected_idx >= len(ph_gallery):
            st.caption("Select an image to enable actions")
        else:
            variant = ph_gallery[selected_idx]
            edit_options = {
                "Relevant Props": "Add relevant props that logically match the product’s use; keep scale realistic and composition balanced.",
                "Packaging": "Add product packaging behind the item, slightly out of focus, matching brand colors.",
                "Hang Tag (blank)": "Add a small premium hang tag near the product; leave the tag blank for later text overlay",
                "Liquid Splash": "Add a dynamic liquid splash element behind the product; keep it subtle and not covering the label.",
                "Levitating": "Place product on a clean pedestal; add soft rim light; make it look slightly levitating.",
                "Surface/Shadow Polish": "Place product on Matte studio surface; refined soft shadow; avoid harsh reflections.",
                "Lifestyle": "Place in a simple relevant lifestyle setting (kitchen/bathroom/desk); shallow depth of field; natural light.",
                "Flat-lay": "top-down composition on a clean backdrop.",
                "Seasonal Hint": "Tasteful seasonal color hint in background only; no decorations.",
            }

            with st.expander("Quick Edit"):
                options = list(edit_options.keys())
                current_choice = st.session_state.get("ph_edit_choice")
                index_value = (
                    options.index(current_choice) if current_choice in options else None
                )
                if index_value is None:
                    st.radio(
                        "Quick Edit",
                        options,
                        key="ph_edit_choice",
                    )
                else:
                    st.radio(
                        "Quick Edit",
                        options,
                        index=index_value,
                        key="ph_edit_choice",
                    )
                if st.session_state.pop("ph_clear_manual", False):
                    st.session_state["ph_edit_manual"] = ""                
                st.text_input(
                    "Manual tweak",
                    key="ph_edit_manual",
                    placeholder="Optional: add a manual tweak",
                )
                if st.button(
                    "Apply",
                    key=f"ph_apply_{selected_idx}",
                    use_container_width=True,
                ):
                    choice = st.session_state.get("ph_edit_choice")
                    if not choice:
                        st.warning("Select a quick edit option first.")
                    else:
                        if len(st.session_state.get("ph_gallery", [])) >= 5:
                            st.warning(
                                "Max 5 images (1 original + 4 edits). Delete or start a new hero to continue."
                            )
                        else:
                            manual = (st.session_state.get("ph_edit_manual") or "").strip()
                            edit_prompt = edit_options[choice]
                            if manual:
                                edit_prompt += "\n" + manual
                            fmt = (variant.get("format") or "png").lower()
                            mime = "image/jpeg" if fmt == "jpg" else "image/png"
                            data_url = "data:%s;base64,%s" % (
                                mime,
                                base64.b64encode(variant["img"]).decode("utf-8"),
                            )
                            try:
                                if variant.get("engine") == "A":
                                    seed_val = random.randint(100000, 999999)
                                    new_bytes = generate_kontext_max(
                                        edit_prompt,
                                        data_url,
                                        aspect_ratio="1:1",
                                        seed=seed_val,
                                        output_format="png",
                                    )
                                    updated_variant = {
                                        "img": new_bytes,
                                        "seed": seed_val,
                                        "engine": "A",
                                        "format": "png",
                                    }
                                else:
                                    new_bytes = generate_nano_banana_replicate(
                                        prompt=edit_prompt,
                                        image_data_urls=[data_url],
                                        aspect_ratio=st.session_state.get(
                                            "ph_ar", "match_input_image"
                                        ),
                                        output_format=st.session_state.get(
                                            "ph_outfmt", "jpg"
                                        ),
                                    )
                                    updated_variant = {
                                        "img": new_bytes,
                                        "seed": None,
                                        "engine": "B",
                                        "format": st.session_state.get(
                                            "ph_outfmt", "jpg"
                                        ),
                                    }
                                st.session_state.ph_gallery.append(updated_variant)
                                st.session_state.ph_selected_idx = (
                                    len(st.session_state.ph_gallery) - 1
                                )
                                st.session_state.ph_clear_manual = True
                                ph_log(
                                    uid,
                                    "image_refine_prompt",
                                    mode="product_hero",
                                    choice=choice,
                                    has_manual=bool(manual),
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ {e}")

            variant = st.session_state.ph_gallery[selected_idx]
            file_ext = (variant.get("format") or "png").lower()
            mime_type = "image/jpeg" if file_ext == "jpg" else "image/png"
            st.download_button(
                "Download",
                data=variant["img"],
                file_name=f"product_hero_image_{selected_idx + 1}.{file_ext}",
                mime=mime_type,
                key=f"ph_download_{selected_idx}",
                use_container_width=True,
            )
            if st.button(
                "Chain Edit",
                key=f"ph_chain_{selected_idx}",
                use_container_width=True,
            ):
                st.session_state.chained_image = variant["img"]
                st.session_state.edit_mode = "Inspire"
                st.session_state.ph_pending_mode = "Inspire"
                st.success(
                    "Image ready in Inspire mode. Switch to 🎨 Image Generator → Inspire to continue."
                )        

# ---- Footer ----
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; font-size: 0.9rem;'>Powered by latest AI models </div>",
    unsafe_allow_html=True
)
