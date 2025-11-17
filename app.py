"""
Simple Smart Recipe Finder (app.py)

A much shorter, easier-to-read version of your app that keeps the same behavior:
- Loads bundled CSV "food_recipessample.csv"
- Lets user enter pantry items (comma/newline separated)
- Lets user pick Course and Diet from dataset values
- Filters by optional max prep/cook time
- Shows only recipes that match at least one pantry item
- Displays recipe name, description, prep/cook time, ingredient list (have/missing), and instructions

Notes:
- This simplified code intentionally removes fuzzy matching and advanced price/wishlist features
  to make the code easier to read and present.
- Keep the CSV file named "food_recipessample.csv" in the same folder as this file.
"""
import re
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smart Recipe Finder (Simple)", page_icon="🍳", layout="wide")

# ---- Small styles ----
st.markdown("""
<style>
  .have {background:#e8f5e9; color:#2e7d32; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;}
  .miss {background:#fff3e0; color:#ef6c00; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;}
</style>
""", unsafe_allow_html=True)

# ---- Helpers (kept minimal) ----
def load_csv(path="food_recipes.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset not found: {path}. Place the CSV next to this script.")
        st.stop()
    return pd.read_csv(path)

def clean_ingredient(text):
    if not isinstance(text, str):
        return ""
    # remove parentheses, leading qty and simple descriptors
    t = re.sub(r'\([^)]*\)', '', text)                      # remove parenthesis
    t = re.sub(r'^[\d\.\-\/\s]+', '', t)                    # leading numbers/qty
    t = re.sub(r'^(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|g|gram|grams|kg|oz|ounce|ounces|lb|pound)\b', '', t, flags=re.I)
    t = re.sub(r'^(chopped|diced|minced|sliced|fresh|grated|shredded)\b', '', t, flags=re.I)
    return t.strip().lower()

def parse_ingredients_cell(cell):
    if pd.isna(cell) or str(cell).strip()=="":
        return []
    # split by commas, semicolons or newlines
    parts = re.split(r'[,\n;|]+', str(cell))
    seen = []
    for p in parts:
        c = clean_ingredient(p)
        if c and c not in seen:
            seen.append(c)
    return seen

def parse_pantry(text):
    if not text:
        return set()
    parts = re.split(r'[,\n]+', text)
    result = set()
    for p in parts:
        name = p.split(":",1)[0].strip().lower()
        if name:
            result.add(name)
    return result

def simple_match(ing, pantry_set):
    # exact or substring token overlap check (simple)
    ing = ing.lower()
    if ing in pantry_set:
        return True
    ing_tokens = set(ing.split())
    for p in pantry_set:
        p_tokens = set(p.split())
        # if half of ingredient tokens appear in pantry token, consider match
        if len(ing_tokens & p_tokens) >= max(1, len(ing_tokens)//2):
            return True
        # substring checks
        if p in ing or ing in p:
            return True
    return False

def score_recipe(ings, pantry_set):
    matched = []
    missing = []
    for ing in ings:
        if simple_match(ing, pantry_set):
            matched.append(ing)
        else:
            missing.append(ing)
    total = len(ings) if ings else 1
    coverage = len(matched) / total
    return len(matched), matched, missing, coverage

def parse_minutes(val):
    # very small helper: return first number in string if present, else NaN
    try:
        s = str(val)
    except:
        return float("nan")
    m = re.search(r'(\d+)', s)
    return float(m.group(1)) if m else float("nan")

# ---- Load and normalize dataset ----
df = load_csv()
if 'recipe_title' in df.columns and 'name' not in df.columns:
    df['name'] = df['recipe_title']
df['description'] = df.get('description', "")
df['ingredients'] = df.get('ingredients', "")
df['instructions'] = df.get('instructions', "")
df['prep_time'] = df.get('prep_time', None).apply(parse_minutes) if 'prep_time' in df.columns else pd.NA
df['cook_time'] = df.get('cook_time', None).apply(parse_minutes) if 'cook_time' in df.columns else pd.NA
df['parsed_ingredients'] = df['ingredients'].apply(parse_ingredients_cell)

# normalize course/diet columns for dropdowns if present
course_col = None
diet_col = None
for c in ['course','course_type','category','recipe_category']:
    if c in df.columns:
        course_col = c
        break
for d in ['diet','diet_type','diet_tag']:
    if d in df.columns:
        diet_col = d
        break

# ---- UI on main page (no sidebar) ----
st.title("🍳 Smart Recipe Finder — Simple")
st.write("Enter what you have and choose Course/Diet to find recipes. Only recipes containing at least one of your items are shown.")

pantry_text = st.text_area("Your pantry (comma or newline separated). Optional quantities ignored.", value="eggs:2\nmilk\nonion")
pantry_set = parse_pantry(pantry_text)
if pantry_set:
    st.markdown("**Detected items:** " + ", ".join(sorted(pantry_set)))
else:
    st.markdown("_No pantry items detected yet._")

st.markdown("---")
cols = st.columns(3)
with cols[0]:
    course_opts = sorted([c for c in df[course_col].dropna().unique()]) if course_col else []
    selected_course = st.selectbox("Course", ["Any"] + course_opts)
with cols[1]:
    diet_opts = sorted([d for d in df[diet_col].dropna().unique()]) if diet_col else []
    selected_diet = st.selectbox("Diet", ["Any"] + diet_opts)
with cols[2]:
    top_n = st.number_input("Top results to show", min_value=1, max_value=100, value=12)

st.markdown("---")
time_cols = st.columns(2)
with time_cols[0]:
    max_prep = st.number_input("Max prep time (min)", min_value=0, value=60)
with time_cols[1]:
    max_cook = st.number_input("Max cook time (min)", min_value=0, value=120)

st.markdown("---")
if st.button("Find recipes"):
    if not pantry_set:
        st.info("Please enter at least one pantry item.")
    else:
        # filter dataset by course/diet/time
        filt = df.copy()
        if selected_course and selected_course != "Any" and course_col:
            filt = filt[filt[course_col].astype(str).str.lower() == str(selected_course).lower()]
        if selected_diet and selected_diet != "Any" and diet_col:
            filt = filt[filt[diet_col].astype(str).str.lower() == str(selected_diet).lower()]
        filt = filt[(filt['prep_time'].fillna(99999) <= float(max_prep)) & (filt['cook_time'].fillna(99999) <= float(max_cook))]

        results = []
        for _, row in filt.iterrows():
            ings = row['parsed_ingredients'] or []
            matched_count, matched, missing, coverage = score_recipe(ings, pantry_set)
            if matched_count < 1:
                continue  # require at least one pantry match
            results.append({
                "name": row.get('name',''),
                "description": row.get('description',''),
                "prep_time": row.get('prep_time'),
                "cook_time": row.get('cook_time'),
                "matched_count": matched_count,
                "matched": matched,
                "missing": missing,
                "coverage": coverage,
                "ingredients": ings,
                "instructions": row.get('instructions','')
            })

        if not results:
            st.info("No matching recipes found (that include at least one of your pantry items).")
        else:
            # sort by matched_count desc, then missing asc, then total time asc
            results = sorted(results, key=lambda r: (-r['matched_count'], r['missing'].__len__(), ( (r['prep_time'] or 0) + (r['cook_time'] or 0) )))
            for r in results[:top_n]:
                st.header(r['name'])
                st.write(f"Prep: {int(r['prep_time']) if pd.notna(r['prep_time']) else 'N/A'} min — Cook: {int(r['cook_time']) if pd.notna(r['cook_time']) else 'N/A'} min")
                st.write(f"Matched: {r['matched_count']}  —  Coverage: {int(r['coverage']*100)}%")
                # ingredients display
                ing_html = []
                for ing in r['ingredients']:
                    if ing in r['matched']:
                        ing_html.append(f"<span class='have'>✅ {ing.title()}</span>")
                    else:
                        ing_html.append(f"<span class='miss'>➕ {ing.title()}</span>")
                st.markdown(" ".join(ing_html), unsafe_allow_html=True)
                if r['missing']:
                    st.warning("Missing: " + ", ".join([m.title() for m in r['missing']]))
                if r['description']:
                    st.info(r['description'])
                if r['instructions']:
                    with st.expander("Show instructions"):
                        st.write(r['instructions'])
                st.markdown("---")

st.caption("This simplified version removes fuzzy matching and wishlist features to make the code shorter and easier to present.")
