# file docstring explaining purpose of the script
import re  # import regex module for string parsing
import os  # import os for file existence checks
import streamlit as st  # import streamlit for the app UI
import pandas as pd  # import pandas for CSV loading and simple data handling

st.set_page_config(page_title="Smart Recipe Finder (Simple)", page_icon="🍳", layout="wide")  # configure Streamlit app page

# ---- Small styles ----
st.markdown("""  # inject small CSS to style ingredient labels
<style>
  .have {background:#e8f5e9; color:#2e7d32; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;}
  .miss {background:#fff3e0; color:#ef6c00; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;}
</style>
""", unsafe_allow_html=True)  # allow raw HTML/CSS in markdown

# ---- Helpers (kept minimal) ----
def load_csv(path="food_recipes.csv"):  # function to load the bundled CSV
    if not os.path.exists(path):  # check file exists
        st.error(f"Dataset not found: {path}. Place the CSV next to this script.")  # show error in UI
        st.stop()  # stop execution if dataset missing
    return pd.read_csv(path)  # return loaded DataFrame

def clean_ingredient(text):  # minimal ingredient cleaning helper
    if not isinstance(text, str):  # if value isn't a string
        return ""  # return empty string for safety
    # remove parentheses, leading qty and simple descriptors
    t = re.sub(r'\([^)]*\)', '', text)                      # remove parenthesis content
    t = re.sub(r'^[\d\.\-\/\s]+', '', t)                    # remove leading numbers/quantities
    t = re.sub(r'^(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|g|gram|grams|kg|oz|ounce|ounces|lb|pound)\b', '', t, flags=re.I)  # remove common unit words
    t = re.sub(r'^(chopped|diced|minced|sliced|fresh|grated|shredded)\b', '', t, flags=re.I)  # strip common prep descriptors
    return t.strip().lower()  # trim and lowercase the cleaned ingredient

def parse_ingredients_cell(cell):  # parse a CSV ingredients cell into a list of normalized items
    if pd.isna(cell) or str(cell).strip()=="":
        return []  # return empty list when cell is empty or NaN
    # split by commas, semicolons or newlines
    parts = re.split(r'[,\n;|]+', str(cell))  # split on common ingredient delimiters
    seen = []  # preserve order while deduplicating
    for p in parts:
        c = clean_ingredient(p)  # clean each part
        if c and c not in seen:
            seen.append(c)  # add unique cleaned ingredient
    return seen  # return list of cleaned unique ingredients

def parse_pantry(text):  # parse user pantry input into a set of names
    if not text:
        return set()  # return empty set if no input
    parts = re.split(r'[,\n]+', text)  # split by comma or newline
    result = set()
    for p in parts:
        name = p.split(":",1)[0].strip().lower()  # ignore optional quantity after colon
        if name:
            result.add(name)  # add normalized name to set
    return result  # return set of pantry item names

def simple_match(ing, pantry_set):  # simple matching rule between recipe ingredient and pantry set
    # exact or substring token overlap check (simple)
    ing = ing.lower()  # normalize ingredient
    if ing in pantry_set:
        return True  # exact match if present in pantry
    ing_tokens = set(ing.split())  # tokens of ingredient name
    for p in pantry_set:
        p_tokens = set(p.split())  # tokens of pantry item
        # if half of ingredient tokens appear in pantry token, consider match
        if len(ing_tokens & p_tokens) >= max(1, len(ing_tokens)//2):
            return True
        # substring checks
        if p in ing or ing in p:
            return True
    return False  # no match found

def score_recipe(ings, pantry_set):  # score recipe vs pantry items
    matched = []  # list of matched ingredients
    missing = []  # list of missing ingredients
    for ing in ings:
        if simple_match(ing, pantry_set):
            matched.append(ing)  # collect matched ingredients
        else:
            missing.append(ing)  # collect missing ingredients
    total = len(ings) if ings else 1  # avoid division by zero
    coverage = len(matched) / total  # fraction of ingredients available
    return len(matched), matched, missing, coverage  # return matched count, matched list, missing list, coverage

def parse_minutes(val):  # lightweight time parser that extracts the first number
    # very small helper: return first number in string if present, else NaN
    try:
        s = str(val)
    except:
        return float("nan")
    m = re.search(r'(\d+)', s)  # find first integer in the string
    return float(m.group(1)) if m else float("nan")  # return as float or NaN

# ---- Load and normalize dataset ----
df = load_csv()  # load the bundled CSV into a DataFrame
if 'recipe_title' in df.columns and 'name' not in df.columns:
    df['name'] = df['recipe_title']  # unify title column name if needed
df['description'] = df.get('description', "")  # ensure description column exists
df['ingredients'] = df.get('ingredients', "")  # ensure ingredients column exists
df['instructions'] = df.get('instructions', "")  # ensure instructions column exists
df['prep_time'] = df.get('prep_time', None).apply(parse_minutes) if 'prep_time' in df.columns else pd.NA  # normalize prep time to number
df['cook_time'] = df.get('cook_time', None).apply(parse_minutes) if 'cook_time' in df.columns else pd.NA  # normalize cook time to number
df['parsed_ingredients'] = df['ingredients'].apply(parse_ingredients_cell)  # parse and normalize ingredients into lists

# normalize course/diet columns for dropdowns if present
course_col = None  # variable to hold which column to use for course
diet_col = None  # variable to hold which column to use for diet
for c in ['course','course_type','category','recipe_category']:  # try common course-like column names
    if c in df.columns:
        course_col = c
        break
for d in ['diet','diet_type','diet_tag']:  # try common diet-like column names
    if d in df.columns:
        diet_col = d
        break

# ---- UI on main page (no sidebar) ----
st.title("🍳 Smart Recipe Finder — Simple")  # show main title
st.write("Enter what you have and choose Course/Diet to find recipes. Only recipes containing at least one of your items are shown.")  # brief instructions

pantry_text = st.text_area("Your pantry (comma or newline separated). Optional quantities ignored.", value="eggs:2\nmilk\nonion")  # input box for pantry text
pantry_set = parse_pantry(pantry_text)  # parse pantry input into a set
if pantry_set:
    st.markdown("**Detected items:** " + ", ".join(sorted(pantry_set)))  # show parsed items
else:
    st.markdown("_No pantry items detected yet._")  # hint if empty

st.markdown("---")  # visual separator
cols = st.columns(3)  # create 3 columns for compact inputs
with cols[0]:
    course_opts = sorted([c for c in df[course_col].dropna().unique()]) if course_col else []  # gather course options if available
    selected_course = st.selectbox("Course", ["Any"] + course_opts)  # course dropdown
with cols[1]:
    diet_opts = sorted([d for d in df[diet_col].dropna().unique()]) if diet_col else []  # gather diet options if available
    selected_diet = st.selectbox("Diet", ["Any"] + diet_opts)  # diet dropdown
with cols[2]:
    top_n = st.number_input("Top results to show", min_value=1, max_value=100, value=12)  # number of results to display

st.markdown("---")  # separator
time_cols = st.columns(2)  # two columns for time filters
with time_cols[0]:
    max_prep = st.number_input("Max prep time (min)", min_value=0, value=60)  # max prep time input
with time_cols[1]:
    max_cook = st.number_input("Max cook time (min)", min_value=0, value=120)  # max cook time input

st.markdown("---")  # separator
if st.button("Find recipes"):  # run search when button pressed
    if not pantry_set:
        st.info("Please enter at least one pantry item.")  # require at least one pantry item
    else:
        # filter dataset by course/diet/time
        filt = df.copy()  # start with full dataset
        if selected_course and selected_course != "Any" and course_col:  # filter by course if selected
            filt = filt[filt[course_col].astype(str).str.lower() == str(selected_course).lower()]
        if selected_diet and selected_diet != "Any" and diet_col:  # filter by diet if selected
            filt = filt[filt[diet_col].astype(str).str.lower() == str(selected_diet).lower()]
        filt = filt[(filt['prep_time'].fillna(99999) <= float(max_prep)) & (filt['cook_time'].fillna(99999) <= float(max_cook))]  # apply time filters

        results = []  # collect matched recipes
        for _, row in filt.iterrows():  # iterate over filtered rows
            ings = row['parsed_ingredients'] or []  # get parsed ingredients list
            matched_count, matched, missing, coverage = score_recipe(ings, pantry_set)  # compute matching info
            if matched_count < 1:
                continue  # require at least one ingredient match to show the recipe
            results.append({
                "name": row.get('name',''),  # recipe name
                "description": row.get('description',''),  # description if present
                "prep_time": row.get('prep_time'),  # prep time numeric
                "cook_time": row.get('cook_time'),  # cook time numeric
                "matched_count": matched_count,  # number of matched ingredients
                "matched": matched,  # list of matched ingredients
                "missing": missing,  # list of missing ingredients
                "coverage": coverage,  # fraction matched
                "ingredients": ings,  # original ingredient list parsed
                "instructions": row.get('instructions','')  # instructions if present
            })

        if not results:
            st.info("No matching recipes found (that include at least one of your pantry items).")  # no matches message
        else:
            # sort by matched_count desc, then missing asc, then total time asc
            results = sorted(results, key=lambda r: (-r['matched_count'], r['missing'].__len__(), ( (r['prep_time'] or 0) + (r['cook_time'] or 0) )))
            for r in results[:top_n]:  # show up to top_n results
                st.header(r['name'])  # recipe title
                st.write(f"Prep: {int(r['prep_time']) if pd.notna(r['prep_time']) else 'N/A'} min — Cook: {int(r['cook_time']) if pd.notna(r['cook_time']) else 'N/A'} min")  # show times
                st.write(f"Matched: {r['matched_count']}  —  Coverage: {int(r['coverage']*100)}%")  # summary of match
                # ingredients display
                ing_html = []  # build html chips for ingredients
                for ing in r['ingredients']:
                    if ing in r['matched']:
                        ing_html.append(f"<span class='have'>✅ {ing.title()}</span>")  # green chip for have
                    else:
                        ing_html.append(f"<span class='miss'>➕ {ing.title()}</span>")  # orange chip for missing
                st.markdown(" ".join(ing_html), unsafe_allow_html=True)  # render ingredient chips as HTML
                if r['missing']:
                    st.warning("Missing: " + ", ".join([m.title() for m in r['missing']]))  # list missing ingredients prominently
                if r['description']:
                    st.info(r['description'])  # show description box
                if r['instructions']:
                    with st.expander("Show instructions"):
                        st.write(r['instructions'])  # show instructions inside an expander
                st.markdown("---")  # separator between recipes

st.caption("This simplified version removes fuzzy matching and wishlist features to make the code shorter and easier to present.")  # footer caption
