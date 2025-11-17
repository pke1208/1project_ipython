
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

def load_dataset(uploaded_file) -> pd.DataFrame:
    # uploaded_file is a file-like object from st.file_uploader or None; fallback to 'food_recipes.csv' in cwd
    if uploaded_file is not None:
        try:
            # uploaded_file is a BytesIO-like; pd.read_csv can read it directly
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()
    else:
        # fallback file name the app expects
        fallback = "food_recipes.csv"
        if os.path.exists(fallback):
            try:
                df = load_recipes_from_path(fallback)
            except Exception as e:
                st.error(f"Error reading {fallback}: {e}")
                st.stop()
        else:
            st.error("No dataset provided. Upload a CSV in the sidebar or place food_recipes.csv in the app folder.")
            st.stop()
    # Validate and normalize columns
    expected = ['name', 'description', 'course', 'diet', 'prep_time', 'cook_time', 'ingredients', 'instructions']
    if not all(col in df.columns for col in expected):
        st.warning(f"CSV doesn't contain all expected columns. Required: {expected}. I will try to continue with available columns.")
    # Coerce times to numeric where possible
    if 'prep_time' in df.columns:
        df['prep_time'] = pd.to_numeric(df['prep_time'], errors='coerce')
    else:
        df['prep_time'] = pd.NA
    if 'cook_time' in df.columns:
        df['cook_time'] = pd.to_numeric(df['cook_time'], errors='coerce')
    else:
        df['cook_time'] = pd.NA
    # Ensure string columns
    for c in ['name', 'description', 'course', 'diet', 'ingredients', 'instructions']:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = ""
    # parse ingredients into list (new column)
    df['parsed_ingredients'] = df['ingredients'].apply(parse_ingredients_cell)
    # Normalize course/diet for dropdown consistency
    df['course_norm'] = df['course'].str.strip().replace('', pd.NA)
    df['diet_norm'] = df['diet'].str.strip().replace('', pd.NA)
    return df

# ---- Sidebar & Main UI ----
# Use markdown (unsafe_allow_html) instead of st.title with unsafe flag
st.markdown('<h1 class="main-title">🍳 Smart Recipe Finder</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tell me what you have, and I’ll suggest recipes that fit your preferences.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Load dataset")
    uploaded = st.file_uploader("Upload recipe CSV (optional)", type=["csv"])
    use_sample_button = st.checkbox("If no CSV, try to use food_recipes.csv in folder", value=True)
    st.markdown("---")
    st.header("Your pantry & preferences")
    ing_text = st.text_area("Ingredients you have (comma or newline separated). Optional ':qty' after item", value="eggs:2\nmilk\nonion", help="Example: eggs:2, milk, flour")
    pantry = parse_pantry_text(ing_text)
    st.write(f"Detected pantry items: {', '.join(sorted(pantry.keys()))}" if pantry else "No pantry items parsed yet.")
    st.markdown("---")
    st.markdown("Filter & matching options")
    fuzz_threshold = st.slider("Fuzzy matching sensitivity (higher = stricter)", 60, 100, 80)
    min_coverage = st.slider("Minimum recipe ingredient coverage required (%)", 0, 100, 0)
    only_fully_makeable = st.checkbox("Show only recipes with no missing ingredients", value=False)
    max_missing = st.number_input("Max missing ingredients allowed (set 0 to disable)", min_value=0, max_value=50, value=10, step=1)
    st.markdown("---")
    st.subheader("Optional: Prices for wishlist estimates")
    st.text_area("Enter per-item price lines (item:price)", value="eggs:0.3\ntomato:0.6\nchicken breast:3.0", height=100, key="price_map_text")
    st.markdown("---")
    st.write("When ready, click:")
    find_btn = st.button("🔍 Find Best Recipes", type="primary")
    st.markdown("---")
    st.write("Tips: adjust fuzzy sensitivity if ingredient names differ (e.g., 'parmesan' vs 'parmigiano').")

if fuzz is None:
    st.warning("fuzzywuzzy not available. Install 'fuzzywuzzy' (and 'python-levenshtein' for speed) for improved string matching: pip install fuzzywuzzy python-levenshtein")

if find_btn:
    # Pass uploaded (file-like) or None; loader will use fallback file if present
    df = load_dataset(uploaded if uploaded is not None else None)
    courses = sorted([c for c in df['course_norm'].dropna().unique()]) if 'course_norm' in df.columns else []
    diets = sorted([d for d in df['diet_norm'].dropna().unique()]) if 'diet_norm' in df.columns else []
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        selected_course = st.selectbox("Course", options=["Any"] + courses, index=0)
    with col_c2:
        selected_diet = st.selectbox("Diet", options=["Any"] + diets, index=0)
    # Time filters with safe defaults
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
        for _, row in filtered.iterrows():
            ingredients = row.get('parsed_ingredients', []) or []
            missing_count, missing_list, coverage = score_recipe(ingredients, pantry, fuzz_threshold=fuzz_threshold)
            est_cost = estimate_missing_cost(missing_list, parse_price_map(st.session_state.get("price_map_text", "")))
            results.append({
                "name": row.get('name', ''),
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
                    if st.button(f"Add '{r['name']}' to wishlist", key=f"wish_{i}"):
                        if "wishlist" not in st.session_state:
                            st.session_state.wishlist = []
                        st.session_state.wishlist.append(r)
                        st.success("Added to wishlist (missing ingredients aggregated in sidebar).")

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
