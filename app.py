import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import io
import csv
import json
import yaml
from fractions import Fraction

# ---- Helper Functions ----
def parse_measurement(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parts = str(value).strip().split()
        if len(parts) == 2:
            whole = float(parts[0])
            frac = float(Fraction(parts[1]))
            return whole + frac
        else:
            return float(Fraction(parts[0]))
    except:
        return None

def calculate_board_feet(length, width, thickness=0.75):
    return (thickness * width * length) / 144

def generate_required_pieces(required_df):
    pieces = []
    for _, row in required_df.iterrows():
        try:
            quantity = int(parse_measurement(row.get('Quantity', 1)))
        except:
            quantity = 1
        length = parse_measurement(row.get('Length'))
        width = parse_measurement(row.get('Width'))
        job = row.get('Job', 'Default Job')
        if length is None or width is None:
            continue
        for _ in range(quantity):
            pieces.append({
                'length': length,
                'width': width,
                'id': f"{length:.3f}x{width:.3f}",
                'job': job
            })
    return sorted(pieces, key=lambda x: max(x['length'], x['width']), reverse=True)

def try_place_pieces(board, pieces, kerf):
    free_rectangles = [{'x': 0, 'y': 0, 'length': board['length'], 'width': board['width']}]
    placements = []
    remaining = []
    for piece in pieces:
        placed = False
        for rect in free_rectangles.copy():
            for rotated in [False, True]:
                p_length = piece['length'] + kerf
                p_width = piece['width'] + kerf
                if rotated:
                    p_length, p_width = p_width, p_length
                if p_length <= rect['length'] and p_width <= rect['width']:
                    placements.append({
                        'piece': piece,
                        'x': rect['x'],
                        'y': rect['y'],
                        'length': p_length - kerf,
                        'width': p_width - kerf,
                        'rotated': rotated
                    })
                    new_rects = [
                        {'x': rect['x'] + p_length, 'y': rect['y'], 'length': rect['length'] - p_length, 'width': p_width},
                        {'x': rect['x'], 'y': rect['y'] + p_width, 'length': rect['length'], 'width': rect['width'] - p_width}
                    ]
                    free_rectangles.remove(rect)
                    free_rectangles.extend([r for r in new_rects if r['length'] > 0 and r['width'] > 0])
                    placed = True
                    break
            if placed:
                break
        if not placed:
            remaining.append(piece)
    return placements, remaining

def optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf):
    pieces = generate_required_pieces(required_df)
    purchased_boards = []
    board_counter = 1
    allowed_lengths_ft = [8, 10, 12]
    allowed_lengths_in = [ft * 12 for ft in allowed_lengths_ft]
    allowed_widths = [4 + 0.5*i for i in range(int((12 - 4) / 0.5) + 1)]
    allowed_boards = []
    for L in allowed_lengths_in:
        for W in allowed_widths:
            allowed_boards.append({
                'length': L,
                'width': W,
                'length_ft': L/12,
                'width_in': W
            })
    while pieces:
        best_utilization = 0
        best_candidate = None
        best_placements = None
        for board in allowed_boards:
            placements, _ = try_place_pieces(board, pieces, kerf)
            if placements:
                used_area = sum(p['length'] * p['width'] for p in placements)
                board_area = board['length'] * board['width']
                utilization = used_area / board_area
                if utilization > best_utilization:
                    best_utilization = utilization
                    best_candidate = board
                    best_placements = placements
        if best_candidate is None:
            st.error("One or more required pieces are too big for any available board option.")
            break
        for placement in best_placements:
            piece = placement['piece']
            if piece in pieces:
                pieces.remove(piece)
        board_bf = calculate_board_feet(best_candidate['length'], best_candidate['width'], thickness)
        purchased_boards.append({
            'board_id': board_counter,
            'board': best_candidate,
            'cuts': best_placements,
            'utilization': best_utilization,
            'board_feet': board_bf
        })
        board_counter += 1
    total_cost = sum(b['board_feet'] * cost_per_bf for b in purchased_boards)
    return purchased_boards, pieces, total_cost

def to_fraction_string(value):
    try:
        frac = Fraction(value).limit_denominator(16)
        if frac.denominator == 1:
            return f"{frac.numerator}"
        else:
            whole = frac.numerator // frac.denominator
            remainder = frac - whole
            if whole > 0 and remainder:
                return f"{whole} {remainder.numerator}/{remainder.denominator}"
            else:
                return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return str(value)

def preview_board_layout(purchased_boards):
    for board in purchased_boards:
        b = board['board']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Board {board['board_id']} ({b['length_ft']:.1f} ft x {b['width_in']:.1f}\" - Utilization: {board['utilization']*100:.1f}%)")
        ax.set_xlim(0, b['length'])
        ax.set_ylim(0, b['width'])
        ax.set_aspect('equal')
        for cut in board['cuts']:
            rect = patches.Rectangle((cut['x'], cut['y']), cut['length'], cut['width'], edgecolor='black', facecolor='lightgrey')
            ax.add_patch(rect)
            label = f"{to_fraction_string(cut['piece']['length'])} x {to_fraction_string(cut['piece']['width'])}\n{cut['piece'].get('job', '')}"
            ax.text(cut['x'] + cut['length']/2, cut['y'] + cut['width']/2, label, ha='center', va='center', fontsize=6, color='red')
        st.pyplot(fig)

def generate_csv(purchased_boards):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Board ID', 'Piece ID', 'Length', 'Width', 'Rotated', 'X', 'Y', 'Job'])
    for board in purchased_boards:
        for cut in board['cuts']:
            writer.writerow([
                board['board_id'],
                cut['piece']['id'],
                f"{cut['length']:.3f}",
                f"{cut['width']:.3f}",
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2),
                cut['piece'].get('job', '')
            ])
    output.seek(0)
    return output.getvalue()

def save_plan_to_json(plan, leftovers, required_df):
    data = {
        'purchase_plan': plan,
        'leftovers': leftovers,
        'required_input': required_df.to_dict(orient='records')
    }
    return json.dumps(data, indent=2)

def load_plan_from_json(json_data):
    data = json.loads(json_data)
    return (
        data['purchase_plan'],
        data.get('leftovers', []),
        pd.DataFrame(data.get('required_input', []))
    )

def save_plan_to_yaml(plan, leftovers, required_df):
    data = {
        'purchase_plan': plan,
        'leftovers': leftovers,
        'required_input': required_df.to_dict(orient='records')
    }
    return yaml.dump(data)

def load_plan_from_yaml(yaml_data):
    data = yaml.safe_load(yaml_data)
    return (
        data['purchase_plan'],
        data.get('leftovers', []),
        pd.DataFrame(data.get('required_input', []))
    )

# ---- Streamlit App UI ----
st.set_page_config(page_title="Lumber Purchase Optimizer", layout="wide")
st.title("📐 Lumber Purchase Optimizer")

if 'job_names' not in st.session_state:
    st.session_state.job_names = set()
if 'required_df' not in st.session_state:
    st.session_state.required_df = pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2, "Job": ""}])

st.sidebar.header("Destination")
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    new_job = st.text_input("Add/Select Job", key="job_name_input")
with col2:
    if st.button("➕") and new_job:
        st.session_state.job_names.add(new_job)
        st.session_state["job_name_input"] = ""
        st.session_state.job_names.add(new_job)
    if st.button("➖") and new_job in st.session_state.job_names:
        st.session_state.job_names.remove(new_job)

st.sidebar.header("Cut & Cost Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.sidebar.header("Plan Persistence Options")
file_format = st.sidebar.radio("Select File Format", options=["JSON", "YAML"])
save_plan_button = st.sidebar.button("Save Plan")
load_file = st.sidebar.file_uploader("Load Plan File", type=["json", "yaml", "yml"])

existing_jobs = set(st.session_state.required_df['Job'].dropna().unique())
job_options = sorted(existing_jobs.union(st.session_state.job_names))
st.session_state.required_df = st.data_editor(
    st.session_state.required_df,
    column_config={
        "Job": st.column_config.SelectboxColumn("Job", options=job_options)
    },
    num_rows="dynamic", use_container_width=True
)

if st.button("🔨 Optimize Lumber Purchase"):
    purchase_plan, leftovers, total_cost = optimize_lumber_purchase(st.session_state.required_df, kerf, thickness, cost_per_bf)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers

    total_board_feet = sum(b['board_feet'] for b in purchase_plan)
    st.success(f"Optimization complete! Total board feet purchased: {total_board_feet:.2f}, Estimated Total Cost: ${total_cost:.2f}")

    preview_board_layout(purchase_plan)

    csv_data = generate_csv(purchase_plan)
    pdf_data = generate_pdf(purchase_plan, leftovers)

    st.download_button("📄 Download CSV", csv_data, file_name="purchase_plan.csv", mime="text/csv")
    st.download_button("📄 Download PDF", pdf_data, file_name="purchase_plan.pdf")
    if leftovers:
        st.warning("Some required pieces could not be allocated to any board. Please review the leftover pieces.")

if save_plan_button and 'purchase_plan' in st.session_state:
    if file_format == "JSON":
        saved_data = save_plan_to_json(
            st.session_state.purchase_plan,
            st.session_state.leftovers,
            st.session_state.required_df
        )
        st.sidebar.download_button("Download JSON", saved_data, file_name="purchase_plan.json", mime="application/json")
    else:
        saved_data = save_plan_to_yaml(
            st.session_state.purchase_plan,
            st.session_state.leftovers,
            st.session_state.required_df
        )
        st.sidebar.download_button("Download YAML", saved_data, file_name="purchase_plan.yaml", mime="text/yaml")

if load_file:
    file_content = load_file.read().decode("utf-8")
    if load_file.name.endswith(".json"):
        purchase_plan, leftovers, required_df = load_plan_from_json(file_content)
    else:
        purchase_plan, leftovers, required_df = load_plan_from_yaml(file_content)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers
    st.session_state.required_df = required_df
    st.experimental_rerun()
