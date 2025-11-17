from typing import List, Dict, Any, Tuple
import re
import os
import math

import streamlit as st
import pandas as pd

# fuzzywuzzy optional
try:
    from fuzzywuzzy import fuzz
except Exception:
    fuzz = None

st.set_page_config(page_title="Smart Recipe Finder - Simple", page_icon="🍳", layout="wide")

st.markdown("""
<style>
    .main-title {font-size: 2.2rem; color: #FF6B6B; text-align: center; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.0rem; color: #555; text-align: center; margin-bottom: 1rem;}
    .ingredient-chip {
        display: inline-block;
        background-color: #e1f5fe;
        color: #0277bd;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem;
        border-radius: 12px;
        font-size: 0.85rem;
    }
    .ingredient-miss { background-color: #fff3e0; color: #ef6c00; }
    .ingredient-have { background-color: #e8f5e9; color: #2e7d32; }
    .match-score {color: #4CAF50; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ---- Helpers ----
UNIT_WORDS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp", "teaspoon", "teaspoons",
    "g", "gram", "grams", "kg", "kilogram", "oz", "ounce", "ounces", "lb", "pound",
    "slice", "slices", "clove", "cloves", "piece", "pieces", "can", "cans", "packet", "package"
}

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).strip().lower())

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
    # dedupe preserving order
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
    # simple substring/token checks
    if pantry:
        for p in pantry.keys():
            p_n = normalize(p)
            if p_n == ing_n:
                return True
            if p_n in ing_n or ing_n in p_n:
                return True
            # token overlap
            ing_tokens = set(ing_n.split())
            p_tokens = set(p_n.split())
            if ing_tokens & p_tokens:
                if len(ing_tokens & p_tokens) / max(1, len(ing_tokens)) >= 0.5:
                    return True
            # fuzzy fallback if available
            if fuzz:
                try:
                    score = fuzz.ratio(ing_n, p_n)
                    if score >= fuzz_threshold:
                        return True
                except Exception:
                    pass
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

def parse_minutes(s: Any) -> float:
    if pd.isna(s):
        return float('nan')
    t = str(s).strip()
    if not t:
        return float('nan')
    t = t.replace("–", "-")
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
    num = re.search(r'(\d+)', t)
    if num:
        return float(num.group(1))
    return float('nan')

# ---- Dataset loader ----
@st.cache_data(ttl=300)
def load_recipes_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(ttl=300)
def load_dataset(fallback_path: str = "food_recipes.csv") -> pd.DataFrame:
    if not os.path.exists(fallback_path):
        st.error(f"Dataset not found: {fallback_path}. Please add the CSV to the app folder.")
        st.stop()
    df = load_recipes_from_path(fallback_path)
    # normalize common columns
    if 'recipe_title' in df.columns and 'name' not in df.columns:
        df['name'] = df['recipe_title']
    if 'description' not in df.columns:
        df['description'] = ""
    if 'ingredients' not in df.columns:
        df['ingredients'] = ""
    if 'instructions' not in df.columns:
        df['instructions'] = ""
    # parse times if present
    if 'prep_time' in df.columns:
        df['prep_time'] = df['prep_time'].apply(parse_minutes)
    else:
        df['prep_time'] = pd.NA
    if 'cook_time' in df.columns:
        df['cook_time'] = df['cook_time'].apply(parse_minutes)
    else:
        df['cook_time'] = pd.NA
    df['parsed_ingredients'] = df['ingredients'].apply(parse_ingredients_cell)
    return df

# ---- Load dataset ----
df = load_dataset()

# ---- UI ----
st.markdown('<h1 class="main-title">🍳 Smart Recipe Finder — Simple</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter what you have and I will show recipes and any missing ingredients.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Your pantry")
    ing_text = st.text_area("List items you have (comma or newline separated). Optional ':qty' after item", value="eggs:2\nmilk\nonion", help="Example: eggs:2, milk, flour")
    st.write("This app will use these items to determine which ingredients are missing for each recipe.")
    find_btn = st.button("🔍 Find Recipes")

if fuzz is None:
    st.warning("Install 'fuzzywuzzy' (and 'python-levenshtein' for speed) for improved matching: pip install fuzzywuzzy python-levenshtein")

pantry = parse_pantry_text(ing_text)
fuzz_threshold = 80  # fixed sensible default

if find_btn:
    if not pantry:
        st.info("Please enter at least one pantry item.")
    else:
        results = []
        for _, row in df.iterrows():
            ingredients = row.get('parsed_ingredients', []) or []
            missing_count, missing_list, coverage = score_recipe(ingredients, pantry, fuzz_threshold=fuzz_threshold)
            results.append({
                "name": row.get('name', '') or row.get('recipe_title', ''),
                "description": row.get('description', ''),
                "prep_time": row.get('prep_time'),
                "cook_time": row.get('cook_time'),
                "ingredients": ingredients,
                "missing_count": missing_count,
                "missing": missing_list,
                "coverage": coverage,
                "instructions": row.get('instructions', '')
            })
        results_df = pd.DataFrame(results)
        # sort by fewest missing, then best coverage, then total time
        results_df['total_time'] = results_df['prep_time'].fillna(0) + results_df['cook_time'].fillna(0)
        results_df.sort_values(by=['missing_count', 'total_time', 'coverage'], ascending=[True, True, False], inplace=True)
        if results_df.empty:
            st.info("No recipes found in the dataset.")
        else:
            top_n = st.number_input("How many top recipes to show", min_value=1, max_value=100, value=12, step=1)
            selected_results = results_df.head(int(top_n)).to_dict(orient='records')
            st.success(f"Showing top {len(selected_results)} recipes (sorted by fewest missing ingredients).")
            cols = st.columns(2)
            for i, r in enumerate(selected_results):
                col = cols[i % 2]
                with col:
                    st.markdown(f"### 🍲 {r['name']}")
                    times = f"Prep: {int(r['prep_time']) if pd.notna(r['prep_time']) else 'N/A'} min  •  Cook: {int(r['cook_time']) if pd.notna(r['cook_time']) else 'N/A'} min"
                    st.write(times)
                    st.write(f"Missing ingredients: **{r['missing_count']}**  •  Coverage: **{int(r['coverage']*100)}%**")
                    # ingredient chips
                    chips_html = ""
                    for ing in r.get('ingredients', []):
                        cls = "ingredient-have" if ingredient_present(ing, pantry, fuzz_threshold=fuzz_threshold) else "ingredient-miss"
                        icon = "✅" if cls == "ingredient-have" else "➕"
                        chips_html += f"<span class='ingredient-chip {cls}'>{icon} {ing.title()}</span>"
                    st.markdown(chips_html, unsafe_allow_html=True)
                    if r.get('missing'):
                        st.warning("Missing: " + ", ".join([m.title() for m in r.get('missing', [])]))
                    if r.get('description'):
                        st.info(r['description'])
                    if r.get('instructions'):
                        with st.expander("Show instructions"):
                            steps = re.split(r'(?<=[\.\?\!])\s+', str(r['instructions']))
                            for idx_step, step in enumerate(steps, 1):
                                if step.strip():
                                    st.write(f"{idx_step}. {step.strip()}")

st.markdown("---")
st.caption("Simple mode: no filters or pricing — just pantry vs recipe ingredients.")
