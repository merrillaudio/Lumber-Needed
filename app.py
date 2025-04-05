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

def calculate_board_feet(length, width, thickness=0.75):
    return (thickness * width * length) / 144

# ---- Convert input into individual cut pieces
def generate_required_pieces(required_df):
    pieces = []
    skipped_rows = 0
    for _, row in required_df.iterrows():
        try:
            quantity = int(parse_measurement(row.get('Quantity', 1)))
            length = parse_measurement(row.get('Length'))
            width = parse_measurement(row.get('Width'))
            project_name = row.get('Project Name', '').strip() or ''
            if length is None or width is None or quantity is None:
                raise ValueError
            for _ in range(quantity):
                pieces.append({
                    'length': length,
                    'width': width,
                    'project_name': project_name,
                    'id': f"{length:.3f}x{width:.3f}"
                })
        except:
            skipped_rows += 1
    return sorted(pieces, key=lambda x: max(x['length'], x['width']), reverse=True), skipped_rows

# ---- Placement Algorithm
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

# ---- Optimization Driver
def optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf):
    pieces, skipped_rows = generate_required_pieces(required_df)
    purchased_boards = []
    board_counter = 1
    allowed_lengths_ft = [8, 10, 12]
    allowed_lengths_in = [ft * 12 for ft in allowed_lengths_ft]
    allowed_widths = [4 + 0.5 * i for i in range(int((12 - 4) / 0.5) + 1)]
    allowed_boards = [{'length': L, 'width': W, 'length_ft': L / 12, 'width_in': W} for L in allowed_lengths_in for W in allowed_widths]

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
        if not best_candidate:
            st.error("❌ Some pieces are too large to fit any board. Please adjust your dimensions.")
            break
        for p in best_placements:
            pieces.remove(p['piece'])
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
    return purchased_boards, pieces, total_cost, skipped_rows

# ---- CSV Export
def generate_csv(purchased_boards, job_name=""):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Job Name', 'Board ID', 'Piece ID', 'Project Name', 'Length', 'Width', 'Rotated', 'X', 'Y'])
    for board in purchased_boards:
        for cut in board['cuts']:
            piece = cut['piece']
            writer.writerow([
                job_name,
                board['board_id'],
                piece['id'],
                piece.get('project_name', ''),
                f"{cut['length']:.3f}",
                f"{cut['width']:.3f}",
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2)
            ])
    return output.getvalue()

# ---- PDF Export
def generate_pdf(purchased_boards, leftovers=None, job_name=""):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for board in purchased_boards:
            b = board['board']
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={'height_ratios': [5, 1]})
            ax1.set_title(f"{job_name} - Board {board['board_id']} - {b['length_ft']:.1f} ft x {b['width_in']:.1f}\" "
                          f"({board['board_feet']:.2f} bf, Utilization: {board['utilization'] * 100:.1f}%)",
                          fontsize=12, color='red')
            ax1.set_xlim(0, b['length'])
            ax1.set_ylim(0, b['width'])
            ax1.set_aspect('equal')

            for cut in board['cuts']:
                ax1.add_patch(patches.Rectangle((cut['x'], cut['y']), cut['length'], cut['width'],
                                                linewidth=1.0, edgecolor='black', facecolor='lightgrey'))
                ax1.text(cut['x'] + cut['length']/2, cut['y'] + cut['width']/2,
                         f"{to_fraction_string(cut['piece']['length'])}\" x {to_fraction_string(cut['piece']['width'])}\"",
                         ha='center', va='center', fontsize=8, color='red')

            ax2.axis('off')
            rows = [f"{cut['piece']['id']:<12} {cut['piece'].get('project_name',''):<15} {to_fraction_string(cut['x']):>6} {to_fraction_string(cut['y']):>6} "
                    f"{to_fraction_string(cut['length']):>8} {to_fraction_string(cut['width']):>8} {'Yes' if cut.get('rotated') else 'No':>8}"
                    for cut in board['cuts']]
            ax2.text(0, 1, "Piece ID     Project Name     X      Y       L       W   Rotated\n" + "-"*65 + "\n" + "\n".join(rows),
                     fontsize=10, family='monospace', va='top')
            pdf.savefig(fig)
            plt.close(fig)

        if leftovers:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.set_title(f"Leftover Pieces - {job_name}", fontsize=12)
            rows = [f"{to_fraction_string(p['length']):>8} {to_fraction_string(p['width']):>8}" for p in leftovers]
            ax.text(0, 1, "Length    Width\n" + "-"*17 + "\n" + "\n".join(rows), fontsize=10, family='monospace', va='top')
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title(f"Summary of Boards Purchased - {job_name}", fontsize=14)
        summary = {}
        for board in purchased_boards:
            dims = (board['board']['length_ft'], board['board']['width_in'])
            summary[dims] = summary.get(dims, 0) + 1
        text_lines = [f"{'Board Dimensions':<20} {'Quantity':>10}", "-" * 32]
        for dims, qty in sorted(summary.items()):
            dims_str = f"{dims[0]:.1f} ft x {dims[1]:.1f}\""
            text_lines.append(f"{dims_str:<20} {qty:>10}")
        ax.text(0.1, 0.9, "\n".join(text_lines), fontsize=12, family='monospace', va='top')
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    return buffer

# ---- JSON/YAML persistence
def save_plan_to_json(plan, leftovers, required_df, job_name=""):
    return json.dumps({
        'job_name': job_name,
        'purchase_plan': plan,
        'leftovers': leftovers,
        'required_input': required_df.to_dict(orient='records')
    }, indent=2)

def load_plan_from_json(json_data):
    data = json.loads(json_data)
    st.session_state["job_name"] = data.get("job_name", "")
    return data['purchase_plan'], data.get('leftovers', []), pd.DataFrame(data.get('required_input', []))

def save_plan_to_yaml(plan, leftovers, required_df, job_name=""):
    return yaml.dump({
        'job_name': job_name,
        'purchase_plan': plan,
        'leftovers': leftovers,
        'required_input': required_df.to_dict(orient='records')
    })

def load_plan_from_yaml(yaml_data):
    data = yaml.safe_load(yaml_data)
    st.session_state["job_name"] = data.get("job_name", "")
    return data['purchase_plan'], data.get('leftovers', []), pd.DataFrame(data.get('required_input', []))

# ---- Streamlit UI
st.set_page_config(page_title="Lumber Purchase Optimizer", layout="wide")
st.title("📐 Lumber Purchase Optimizer")

job_name = st.text_input("Job Name", value=st.session_state.get("job_name", ""))
st.session_state["job_name"] = job_name

st.sidebar.header("Cut & Cost Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.sidebar.header("Plan Persistence Options")
file_format = st.sidebar.radio("Select File Format", options=["JSON", "YAML"])
save_plan_button = st.sidebar.button("Save Plan")
load_file = st.sidebar.file_uploader("Load Plan File", type=["json", "yaml", "yml"])

st.subheader("Required Cuts")
def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2, "Project Name": ""}])
required_df = st.data_editor(st.session_state.get('required_df', default_cut_df()), num_rows="dynamic", use_container_width=True)

# ---- Optimization Trigger
if st.button("🔨 Optimize Lumber Purchase"):
    purchase_plan, leftovers, total_cost, skipped_rows = optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers
    st.session_state.required_df = required_df

    total_board_feet = sum(b['board_feet'] for b in purchase_plan)
    st.success(f"Optimization complete! Total board feet purchased: {total_board_feet:.2f}, Estimated Total Cost: ${total_cost:.2f}")

    if skipped_rows:
        st.warning(f"⚠️ {skipped_rows} row(s) were skipped due to missing or invalid Length, Width, or Quantity.")

    csv_data = generate_csv(purchase_plan, job_name=job_name)
    pdf_data = generate_pdf(purchase_plan, leftovers, job_name=job_name)

    st.download_button("📄 Download CSV", csv_data, file_name=f"{job_name or 'purchase_plan'}.csv", mime="text/csv")
    st.download_button("📄 Download PDF", pdf_data, file_name=f"{job_name or 'purchase_plan'}.pdf")

    st.subheader("📏 Board Layout Previews")
    for board in purchase_plan:
        b = board['board']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"{job_name} - Board {board['board_id']} ({b['length_ft']:.1f} ft x {b['width_in']:.1f}\", Utilization: {board['utilization'] * 100:.1f}%)", fontsize=12, color='red')
        ax.set_xlim(0, b['length'])
        ax.set_ylim(0, b['width'])
        ax.set_aspect('equal')
        ax.set_xlabel("Length (in)")
        ax.set_ylabel("Width (in)")
        for cut in board['cuts']:
            rect = patches.Rectangle((cut['x'], cut['y']), cut['length'], cut['width'], linewidth=1.0, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            label = f"{to_fraction_string(cut['piece']['length'])}\" x {to_fraction_string(cut['piece']['width'])}\""
            if cut['piece'].get('project_name'):
                label += f"\n({cut['piece']['project_name']})"
            ax.text(cut['x'] + cut['length'] / 2, cut['y'] + cut['width'] / 2, label, ha='center', va='center', fontsize=8)
        st.pyplot(fig)

    if leftovers:
        st.warning("Some required pieces could not be allocated to any board. Please review the leftover pieces.")

# ---- Save/Load Buttons
if save_plan_button and 'purchase_plan' in st.session_state:
    if file_format == "JSON":
        saved_data = save_plan_to_json(st.session_state.purchase_plan, st.session_state.leftovers, st.session_state.required_df, job_name)
        st.sidebar.download_button("Download JSON", saved_data, file_name=f"{job_name or 'purchase_plan'}.json", mime="application/json")
    else:
        saved_data = save_plan_to_yaml(st.session_state.purchase_plan, st.session_state.leftovers, st.session_state.required_df, job_name)
        st.sidebar.download_button("Download YAML", saved_data, file_name=f"{job_name or 'purchase_plan'}.yaml", mime="text/yaml")

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
