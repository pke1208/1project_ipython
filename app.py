from typing import List, Dict, Any, Tuple
import re
import os
import math
import io

import streamlit as st
import pandas as pd

# fuzzywuzzy may be slower without python-levenshtein; try to import and warn
try:
    from fuzzywuzzy import fuzz
except Exception:
    fuzz = None

st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍳", layout="wide")

# ---- Styles ----
st.markdown("""
<style>
    .main-title {font-size: 2.4rem; color: #FF6B6B; text-align: center; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.0rem; color: #555; text-align: center; margin-bottom: 1rem;}
    .ingredient-chip {
        display: inline-block;
        background-color: #e1f5fe;
        color: #0277bd;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        border-radius: 14px;
        font-size: 0.85rem;
    }
    .ingredient-miss { background-color: #fff3e0; color: #ef6c00; }
    .ingredient-have { background-color: #e8f5e9; color: #2e7d32; }
    .match-score {color: #4CAF50; font-weight: bold;}
    .stButton>button {background-color: #FF6B6B; color: white; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ---- Helpers ----
UNIT_WORDS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "g", "gram", "grams", "kg", "kilogram", "oz", "ounce", "ounces", "lb", "pound",
    "slice", "slices", "clove", "cloves", "piece", "pieces", "can", "cans", "packet", "package"
}

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())

def strip_parentheses(text: str) -> str:
    return re.sub(r'\([^)]*\)', '', text)

def remove_leading_qty_unit(text: str) -> str:
    t = text.strip()
    t = re.sub(r'^[\d\.\-\/\s]+', '', t)
    parts = t.split()
    i = 0
    while i < len(parts):
        p = parts[i].lower().rstrip('.,')
        if p in UNIT_WORDS:
            i += 1
        else:
            break
    return ' '.join(parts[i:]).strip()

def parse_ingredients_cell(cell: Any) -> List[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        tokens = list(cell)
    else:
        tokens = re.split(r'[,\n;\|]+', str(cell))
    out = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        tok = strip_parentheses(tok)
        tok = remove_leading_qty_unit(tok)
        tok = re.sub(r'^(chopped|diced|minced|sliced|fresh|grated|shredded|room temperature)\s+', '', tok, flags=re.I)
        cleaned = normalize(tok)
        if cleaned:
            out.append(cleaned)
    seen = set()
    res = []
    for i in out:
        if i not in seen:
            seen.add(i)
            res.append(i)
    return res

def parse_pantry_text(pantry_text: str) -> Dict[str, float]:
    pantry: Dict[str, float] = {}
    if not pantry_text:
        return pantry
    lines = []
    if '\n' in pantry_text:
        lines = pantry_text.splitlines()
    else:
        lines = re.split(r',\s*', pantry_text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, qty = [p.strip() for p in line.split(":", 1)]
            try:
                q = float(qty)
            except Exception:
                q = math.inf
        else:
            name = line
            q = math.inf
        pantry[normalize(name)] = q
    return pantry

def ingredient_present(ing: str, pantry: Dict[str, float], fuzz_threshold: int = 80) -> bool:
    ing_n = normalize(ing)
    if ing_n in pantry:
        return True
    ing_tokens = set(ing_n.split())
    for p in pantry.keys():
        p_n = normalize(p)
        if p_n == ing_n:
            return True
        if p_n in ing_n or ing_n in p_n:
            return True
        if fuzz:
            try:
                score = fuzz.ratio(ing_n, p_n)
                if score >= fuzz_threshold:
                    return True
            except Exception:
                pass
        p_tokens = set(p_n.split())
        if ing_tokens & p_tokens:
            if len(ing_tokens & p_tokens) / max(1, len(ing_tokens)) >= 0.5:
                return True
    return False

def score_recipe(ingredients: List[str], pantry: Dict[str, float], fuzz_threshold: int) -> Tuple[int, List[str], float]:
    missing = []
    matched = 0
    for ing in ingredients:
        if ingredient_present(ing, pantry, fuzz_threshold=fuzz_threshold):
            matched += 1
        else:
            missing.append(ing)
    total = len(ingredients) if ingredients else 1
    coverage = matched / total
    return len(missing), missing, coverage

def parse_price_map(text: str) -> Dict[str, float]:
    price_map = {}
    if not text:
        return price_map
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if ":" in line:
            name, val = [p.strip() for p in line.split(":", 1)]
            try:
                price_map[normalize(name)] = float(val)
            except Exception:
                continue
    return price_map

def estimate_missing_cost(missing: List[str], price_map: Dict[str, float]) -> float:
    total = 0.0
    for ing in missing:
        p = price_map.get(ing)
        if p is None:
            for k in price_map.keys():
                if k in ing or ing in k:
                    p = price_map[k]
                    break
        if p:
            total += p
    return round(total, 2)

def aggregate_wishlist(selected: List[Dict[str, Any]], price_map: Dict[str, float]) -> List[Dict[str, Any]]:
    agg = {}
    for r in selected:
        for ing in r.get("missing", []):
            agg[ing] = agg.get(ing, 0) + 1
    out = []
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        price = price_map.get(k)
        if price is None:
            for pk in price_map.keys():
                if pk in k or k in pk:
                    price = price_map[pk]
                    break
        out.append({"item": k, "count": v, "estimated_unit_price": price if price is not None else "", "estimated_total": round((price or 0) * v, 2)})
    return out

# ---- Load dataset ----
@st.cache_data(ttl=300)
def load_recipes_from_path(path: str) -> pd.DataFrame:
    # read the CSV path provided
    df = pd.read_csv(path)
    return df

def parse_minutes(s: Any) -> float:
    if pd.isna(s):
        return float('nan')
    t = str(s).strip()
    if not t:
        return float('nan')
    t = t.replace("–", "-")
    # try hours and minutes
    hours = 0
    minutes = 0
    h_matches = re.findall(r'(\d+)\s*[hH]', t)
    m_matches = re.findall(r'(\d+)\s*[mM]', t)
    if h_matches:
        hours = sum(int(x) for x in h_matches)
    if m_matches:
        minutes = sum(int(x) for x in m_matches)
    if h_matches or m_matches:
        return hours * 60 + minutes
    # fallback: extract first number (assume minutes)
    num = re.search(r'(\d+)', t)
    if num:
        return float(num.group(1))
    return float('nan')

@st.cache_data(ttl=300)
def load_dataset(fallback_path: str = "food_recipessample.csv") -> pd.DataFrame:
    if os.path.exists(fallback_path):
        try:
            df = load_recipes_from_path(fallback_path)
        except Exception as e:
            st.error(f"Error reading dataset at {fallback_path}: {e}")
            st.stop()
    else:
        st.error(f"Dataset not found: {fallback_path}. Please ensure the sample CSV is in the app folder as '{fallback_path}'.")
        st.stop()

    # Map known alternate column names from sample to the app's expected names
    # Many sample files use 'recipe_title' for name
    if 'recipe_title' in df.columns and 'name' not in df.columns:
        df['name'] = df['recipe_title']
    # keep description if available
    if 'description' not in df.columns:
        df['description'] = ""
    # course/diet columns
    if 'course' not in df.columns:
        df['course'] = ""
    if 'diet' not in df.columns:
        df['diet'] = ""
    # times - try to parse flexible time strings like "15 M" or "1 H 30 M"
    if 'prep_time' in df.columns:
        df['prep_time'] = df['prep_time'].apply(parse_minutes)
    else:
        df['prep_time'] = pd.NA
    if 'cook_time' in df.columns:
        df['cook_time'] = df['cook_time'].apply(parse_minutes)
    else:
        df['cook_time'] = pd.NA
    # Ensure ingredients and instructions exist
    if 'ingredients' not in df.columns:
        df['ingredients'] = ""
    if 'instructions' not in df.columns:
        df['instructions'] = ""
    # parse ingredients into list
    df['parsed_ingredients'] = df['ingredients'].apply(parse_ingredients_cell)
    df['course_norm'] = df['course'].astype(str).str.strip().replace('', pd.NA)
    df['diet_norm'] = df['diet'].astype(str).str.strip().replace('', pd.NA)
    return df

# ---- Load the dataset once (no upload required) ----
df = load_dataset()

# ---- Sidebar & Main UI ----
st.markdown('<h1 class="main-title">🍳 Smart Recipe Finder</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tell me what you have, choose a course & diet, and I’ll suggest recipes from the bundled dataset.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Your pantry & preferences")
    ing_text = st.text_area("Ingredients you have (comma or newline separated). Optional ':qty' after item", value="eggs:2\nmilk\nonion", help="Example: eggs:2, milk, flour")
    pantry = parse_pantry_text(ing_text)
    st.write(f"Detected pantry items: {', '.join(sorted(pantry.keys()))}" if pantry else "No pantry items parsed yet.")
    st.markdown("---")

    # Removed "Filter & matching options" UI per request.
    # Set defaults so behavior remains consistent.
    # fuzz_threshold slider removed -> default value:
    # min_coverage slider removed -> default 0
    # only_fully_makeable checkbox removed -> default False
    # max_missing number_input removed -> default 10
    # Also removed the "Optional: Prices for wishlist estimates" text area UI.

    # Default matching/filtering values (previous defaults preserved)
    # These are here so subsequent logic that references them continues to work.
    fuzz_threshold = 80
    min_coverage = 0
    only_fully_makeable = False
    max_missing = 10

    st.write("When ready, click:")
    find_btn = st.button("🔍 Find Best Recipes", type="primary")
    st.markdown("---")
    st.write("Tips: adjust fuzzy sensitivity if ingredient names differ (e.g., 'parmesan' vs 'parmigiano').")

if fuzz is None:
    st.warning("fuzzywuzzy not available. Install 'fuzzywuzzy' (and 'python-levenshtein' for speed) for improved string matching: pip install fuzzywuzzy python-levenshtein")

# Course & Diet selectors using dataset values
courses = sorted([c for c in df['course_norm'].dropna().unique()]) if 'course_norm' in df.columns else []
diets = sorted([d for d in df['diet_norm'].dropna().unique()]) if 'diet_norm' in df.columns else []
col_c1, col_c2 = st.columns(2)
with col_c1:
    selected_course = st.selectbox("Course", options=["Any"] + courses, index=0)
with col_c2:
    selected_diet = st.selectbox("Diet", options=["Any"] + diets, index=0)

# Time filters with safe defaults derived from dataset
min_prep = int(df['prep_time'].dropna().min()) if df['prep_time'].dropna().size else 0
max_prep_avail = int(df['prep_time'].dropna().max()) if df['prep_time'].dropna().size else 120
min_cook = int(df['cook_time'].dropna().min()) if df['cook_time'].dropna().size else 0
max_cook_avail = int(df['cook_time'].dropna().max()) if df['cook_time'].dropna().size else 240
col_t1, col_t2 = st.columns(2)
with col_t1:
    max_prep_allowed = st.number_input("Max prep time (minutes)", min_value=0, max_value=max(240, max_prep_avail), value=max(30, min_prep))
with col_t2:
    max_cook_allowed = st.number_input("Max cook time (minutes)", min_value=0, max_value=max(1440, max_cook_avail), value=max(60, min_cook))
st.markdown("---")

# Main search/run logic
if find_btn:
    filtered = df.copy()
    if selected_course != "Any":
        filtered = filtered[filtered['course_norm'].str.lower() == selected_course.lower()]
    if selected_diet != "Any":
        filtered = filtered[filtered['diet_norm'].str.lower() == selected_diet.lower()]

    filtered = filtered[(filtered['prep_time'].fillna(99999) <= float(max_prep_allowed)) & (filtered['cook_time'].fillna(99999) <= float(max_cook_allowed))]
    if filtered.empty:
        st.info("No recipes match your course/diet/time filters. Try relaxing some constraints.")
    else:
        results = []
        price_map = parse_price_map(st.session_state.get("price_map_text", ""))  # text input removed; key will default to ""
        for _, row in filtered.iterrows():
            ingredients = row.get('parsed_ingredients', []) or []
            missing_count, missing_list, coverage = score_recipe(ingredients, pantry, fuzz_threshold=fuzz_threshold)
            est_cost = estimate_missing_cost(missing_list, price_map)
            results.append({
                "name": row.get('name', '') or row.get('recipe_title', ''),
                "description": row.get('description', ''),
                "course": row.get('course', ''),
                "diet": row.get('diet', ''),
                "prep_time": row.get('prep_time'),
                "cook_time": row.get('cook_time'),
                "ingredients": ingredients,
                "missing_count": missing_count,
                "missing": missing_list,
                "coverage": coverage,
                "estimated_missing_cost": est_cost,
                "instructions": row.get('instructions', '')
            })
        results_df = pd.DataFrame(results)
        if only_fully_makeable:
            results_df = results_df[results_df['missing_count'] == 0]
        if min_coverage > 0:
            results_df = results_df[results_df['coverage'] >= (min_coverage / 100.0)]
        if max_missing > 0:
            results_df = results_df[results_df['missing_count'] <= max_missing]
        if results_df.empty:
            st.info("No recommendations after applying matching filters. Try lowering min coverage, increasing max missing, or relaxing times.")
        else:
            results_df = results_df.copy()
            results_df['total_time'] = results_df['prep_time'].fillna(0) + results_df['cook_time'].fillna(0)
            results_df.sort_values(by=['missing_count', 'total_time', 'coverage'], inplace=True)
            results_df['score'] = results_df.apply(lambda r: round(r['coverage']*100 - r['missing_count']*5, 1), axis=1)
            top_n = st.number_input("How many top recipes to show", min_value=1, max_value=50, value=6, step=1)
            selected_results = results_df.head(int(top_n)).to_dict(orient='records')

            st.success(f"Found {len(results_df)} candidate recipes — showing top {len(selected_results)}")
            cols = st.columns(2)
            for i, r in enumerate(selected_results):
                col = cols[i % 2]
                with col:
                    st.markdown(f"### 🍲 {r['name']}  —  <span class='match-score'>{r['score']}</span>", unsafe_allow_html=True)
                    meta = f"Course: {r.get('course','-')}  |  Diet: {r.get('diet','-')}"
                    times = f"Prep: {int(r['prep_time']) if pd.notna(r['prep_time']) else 'N/A'} min  •  Cook: {int(r['cook_time']) if pd.notna(r['cook_time']) else 'N/A'} min"
                    st.write(meta)
                    st.write(times)
                    st.write(f"Missing ingredients: **{r['missing_count']}**  •  Coverage: **{int(r['coverage']*100)}%**  •  Est. missing cost: **${r['estimated_missing_cost']}**")
                    chips_html = ""
                    for ing in r.get('ingredients', []):
                        cls = "ingredient-have" if ingredient_present(ing, pantry, fuzz_threshold=fuzz_threshold) else "ingredient-miss"
                        human = ing.title()
                        icon = "✅" if cls == "ingredient-have" else "➕"
                        chips_html += f"<span class='ingredient-chip {cls}'>{icon} {human}</span>"
                    st.markdown(chips_html, unsafe_allow_html=True)
                    if r.get('description'):
                        st.info(r['description'])
                    if r.get('instructions'):
                        with st.expander("Show instructions"):
                            steps = re.split(r'(?<=[\.\?\!])\s+', r['instructions'])
                            for idx_step, step in enumerate(steps, 1):
                                if step.strip():
                                    st.write(f"{idx_step}. {step.strip()}")

# ---- Sidebar wishlist display & download ----
st.sidebar.markdown("---")
st.sidebar.subheader("Wishlist & Shopping")
if "wishlist" in st.session_state and st.session_state.wishlist:
    price_map = parse_price_map(st.session_state.get("price_map_text", ""))
    agg = aggregate_wishlist(st.session_state.wishlist, price_map)
    wdf = pd.DataFrame(agg)
    st.sidebar.table(wdf if not wdf.empty else pd.DataFrame([{"item":"(empty)","count":0}]))
    csv_buf = io.StringIO()
    if not wdf.empty:
        wdf.to_csv(csv_buf, index=False)
        st.sidebar.download_button("Download wishlist CSV", data=csv_buf.getvalue(), file_name="wishlist.csv", mime="text/csv")
    if st.sidebar.button("Clear wishlist"):
        st.session_state.wishlist = []
        st.sidebar.success("Wishlist cleared.")
else:
    st.sidebar.write("No items in wishlist yet. Add from search results.")

# ---- Footer ----
st.markdown("---")
st.caption("Built for intermediate Python • Improved matching, filters, wishlist & cost estimates.")
