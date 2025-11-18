# AI Content Builder

AI Content Builder is a Streamlit application that helps marketers and creative teams generate on-brand copy, images, and shoppable product hero shots from a single interface. After authenticating, teammates can switch between a text generator for OTT and social messages, a multi-engine image lab (create, inspire/edit, combine, quick actions), and a Product Hero workspace for background cleanup and prop styling. Admin users get a dedicated dashboard for quotas, event logs, and storage cleanup via Supabase.

## Key capabilities

- **Role-based access + admin console** – Users authenticate against the Supabase `users` table. Admins land in a dashboard that surfaces quota usage, success/error metrics, event timelines, and stored image galleries before offering purge/reset tools. Non-admins are routed into the creative workspace tabs.
- **Text generator for OTT + social** – The first tab lets you toggle between OTT/Direct Messaging (WhatsApp/SMS/Viber) and social posts (LinkedIn/Facebook/Viva Engage). Social posts can optionally ingest up to four “Copy my style” samples to build a style card that conditions the OpenAI `gpt-4o-mini` prompts before generating multiple variants, logging outputs, and letting the user refine selections.
- **Image lab with four modes** – The image tab supports Create (Flux Schnell / Gemini Nano Banana text-to-image), Inspire (single-image edits with vectorization and templated prompts), Combine Images (multi-upload composition), and Quick Actions (background removal, overlays, color swaps, resize/crop utilities) while enforcing daily quotas and logging every action.
- **Product Hero workspace** – Upload a product cutout, build a palette, and generate refined e-commerce hero shots. Quick edit recipes can branch new variants, download assets, or chain edits back into Inspire mode. Product Hero logging can be toggled via secrets.

## Prerequisites

- **Python**: 3.10+ is recommended for compatibility with Streamlit and the Supabase Python client.
- **Streamlit** and all dependencies listed in [`requirements.txt`](requirements.txt). Install them with `pip install -r requirements.txt`.
- **Accounts & quotas**: Active OpenAI, Replicate, and Supabase projects with the right API quotas; optional Product Hero logging toggle.

## Local setup

1. **Clone the repository** and create a virtual environment targeting Python 3.10+.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create `.streamlit/secrets.toml`** (see [Configuration](#configuration) below) so Streamlit can load API keys, Supabase info, and quota settings.
4. **Run the app**:
   ```bash
   streamlit run ai_content_build.py
   ```
5. Visit `http://localhost:8501` and sign in with a Supabase user that exists in the `users` table (or the bootstrapped admin).

## Configuration

The app relies on Streamlit secrets for every external integration. Create a `.streamlit/secrets.toml` file with entries like the following:

```toml
OPENAI_API_KEY = "sk-live..."
REPLICATE_API_TOKEN = "r8_..."
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE = "eyJhbGci..."
BOOTSTRAP_ADMIN_USERNAME = "admin"
BOOTSTRAP_ADMIN_PASSWORD = "change-me"
GLOBAL_IMAGE_QUOTA = 10
EXCLUDE_FROM_QUOTA_MODES = "text_overlay,resize_crop"
DISABLE_PRODUCT_HERO_LOGS = false
```

Key details:

- **OpenAI** powers prompt refinement, text generation, and style card building.
- **Replicate** handles Flux Schnell, Flux Kontext Max, Nano Banana, Recraft Vectorize, and related workflows (`REPLICATE_API_TOKEN`).
- **Supabase** requires the project URL plus the service role key to use tables/storage. Images are saved to an `images` storage bucket and mirrored to the `images` table, while `users` and `events` tables back authentication and analytics.
- **Bootstrap admin credentials** – If no admin exists, the app hashes `BOOTSTRAP_ADMIN_USERNAME` / `BOOTSTRAP_ADMIN_PASSWORD` into Supabase automatically.
- **Quotas & logging** – `GLOBAL_IMAGE_QUOTA` caps per-user image generations per day, and `EXCLUDE_FROM_QUOTA_MODES` removes utility modes (vectorize, text overlay, resize/crop, etc.) from the count. Product Hero logging can be suppressed by flipping `DISABLE_PRODUCT_HERO_LOGS`.

Store secrets securely; avoid committing the TOML file to Git.

## Preparing Supabase

1. **Create tables** matching the schema implied in `ai_content_build.py`:
   - `users` with `id`, `username`, `password_hash`, and `role` for auth.
   - `events` for usage logs/quota tracking (fields referenced in `fetch_events` and `log_event`).
   - `images` for metadata shown in the admin gallery.
2. **Create a storage bucket** named `images` and expose public URLs for generated files.
3. **Set Row Level Security** rules so the service role can read/write as required by the app.

## Running on Streamlit Community Cloud

1. Push this repository to GitHub (or another git host that Streamlit Cloud can access).
2. Create a new Streamlit app and select `ai_content_build.py` as the entrypoint.
3. In the Streamlit Cloud dashboard, open **App settings → Secrets** and paste the same TOML block you use locally.
4. Specify Python 3.10 in **App settings → Advanced** if needed (Streamlit Cloud detects it from `requirements.txt`, but pinning removes ambiguity).
5. Redeploy whenever you update the repo; the single-file Streamlit app will reload automatically.

## Launch checklist

- ✅ Dependencies installed from `requirements.txt`.
- ✅ `.streamlit/secrets.toml` contains all required API keys and Supabase settings.
- ✅ Supabase tables/storage exist and service role credentials are set.
- ✅ Streamlit app runs locally (`streamlit run ai_content_build.py`).
- ✅ Secrets replicated in Streamlit Cloud (if deploying) before inviting teammates.

Once deployed, invite marketers to create Supabase user accounts, monitor usage through the admin tab, and iterate on prompts via version control.
