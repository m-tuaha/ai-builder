Project

AI Content Builder (Streamlit, single-file: ai_content_build.py). Uses OpenAI (text), Replicate [flux kontext-max as Model A and Google's Nano-Banana as MOdel B] (image), Supabase (storage + logs). Roles: tester (Text & Image), admin (Admin only; cannot generate). Times in Admin = Asia/Karachi, DB = UTC.

What you (the agent) should do

Generate unified diff versions for ai_content_build.py modification only unless explicitly told otherwise (ASK mode).

Keep changes surgical; no schema changes and no new deps unless explicitly requested.

Follow logging/quotas/retention already in code and Supabase schema (see main/prompts/supabase_schema_sql.txt).

Respect OTT vs Social rules in prompts (text_system.md, image_system.md, social_system.md, style_social.md).


Never print secrets to logs.

Files to read first:

ai_content_build.py (MOST IMPORTANT FILE)

Other files: 
main/data/supabase_schema_sql.txt, main/prompts/image_system.md, main/prompts/inspire_templates.json, main/prompts/text_system.md
