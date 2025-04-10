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
        project = row.get('Project Name', '')
        if length is None or width is None:
            continue
        for _ in range(quantity):
            pieces.append({
                'length': length,
                'width': width,
                'id': f"{length:.3f}x{width:.3f}",
                'project': project
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
    allowed_boards = [{'length': L, 'width': W, 'length_ft': L/12, 'width_in': W} for L in allowed_lengths_in for W in allowed_widths]
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

def generate_pdf(purchased_boards, leftovers=None, job_title=""):
    buffer = io.BytesIO()
    page_width, page_height = 8.5, 11
    with PdfPages(buffer) as pdf:

        # Draw individual board pages (unchanged)
        for board in purchased_boards:
            b = board['board']
            fig = plt.figure(figsize=(page_width, page_height))
            gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.3)
            ax_draw = fig.add_subplot(gs[0])
            board_title = f"Board {board['board_id']} - {b['length_ft']:.1f} ft x {b['width_in']:.1f}\" ({board['board_feet']:.2f} bf, Utilization: {board['utilization']*100:.1f}%)"
            ax_draw.set_title(board_title, fontsize=12, color='red')
            ax_draw.set_xlabel("Inches", fontsize=10)
            ax_draw.set_ylabel("Inches", fontsize=10)
            ax_draw.set_xlim(0, b['length'])
            ax_draw.set_ylim(0, b['width'])
            ax_draw.set_aspect('equal')
            ax_draw.text(0, b['width'] + 2, f"Job: {job_title}", fontsize=10, color='blue')
            for cut in board['cuts']:
                rect = patches.Rectangle((cut['x'], cut['y']), cut['length'], cut['width'], edgecolor='black', facecolor='lightgrey')
                ax_draw.add_patch(rect)
                label = f"{to_fraction_string(cut['piece']['length'])}\" x {to_fraction_string(cut['piece']['width'])}\""
                ax_draw.text(cut['x'] + cut['length']/2, cut['y'] + cut['width']/2, label, ha='center', va='center', fontsize=8, color='red')
            ax_text = fig.add_subplot(gs[1])
            ax_text.axis('off')
            lines = [f"{'Piece ID':<12} {'X':>6} {'Y':>6} {'L':>8} {'W':>8} {'Rotated':>8}", "-" * 50]
            for cut in board['cuts']:
                lines.append(f"{cut['piece']['id']:<12} {to_fraction_string(cut['x']):>6} {to_fraction_string(cut['y']):>6} {to_fraction_string(cut['length']):>8} {to_fraction_string(cut['width']):>8} {'Yes' if cut.get('rotated') else 'No':>8}")
            ax_text.text(0, 1, "\n".join(lines), fontsize=10, family='monospace', va='top')
            pdf.savefig(fig)
            plt.close(fig)

        # Build the summary information
        summary = {}
        for board in purchased_boards:
            for cut in board['cuts']:
                project = cut['piece'].get('project', 'Unknown')
                bf = calculate_board_feet(cut['piece']['length'], cut['piece']['width'])
                summary[project] = summary.get(project, 0) + bf

        # Aggregate boards by dimensions
        board_summary = {}
        for board in purchased_boards:
            b = board['board']
            key = f"{b['length_ft']:.1f} ft x {b['width_in']:.1f}\""
            board_summary[key] = board_summary.get(key, 0) + 1

        # Build a list of summary lines
        summary_lines = []
        summary_lines.append("Lumber Summary by Project")
        summary_lines.append("")  # blank line for spacing
        for project, bf in summary.items():
            summary_lines.append(f"{project}: {bf:.2f} board feet")
        summary_lines.append("")
        summary_lines.append("Boards to Purchase:")
        for board in purchased_boards:
            b = board['board']
            summary_lines.append(f"Board {board['board_id']}: {b['length_ft']:.1f} ft x {b['width_in']:.1f}\" ({board['board_feet']:.2f} bf)")
        summary_lines.append("")
        summary_lines.append("Aggregate Board Purchase Requirements:")
        for key, count in board_summary.items():
            summary_lines.append(f"{key}: {count}")

        # Define layout parameters for the summary text
        top_y = 0.9        # starting vertical position
        bottom_y = 0.1     # bottom margin (if needed)
        line_height = 0.03 # vertical spacing between lines
        lines_per_page = int((top_y - bottom_y) // line_height)

        # Loop over the summary lines in chunks, creating a new PDF page for each chunk
        for i in range(0, len(summary_lines), lines_per_page):
            fig_summary = plt.figure(figsize=(page_width, page_height))
            ax = fig_summary.add_subplot(111)
            ax.axis('off')
            page_chunk = summary_lines[i:i+lines_per_page]
            y = top_y
            for line in page_chunk:
                ax.text(0.1, y, line, fontsize=12, va='top')
                y -= line_height
            pdf.savefig(fig_summary)
            plt.close(fig_summary)
    buffer.seek(0)
    return buffer

def generate_csv(purchased_boards, job_title=""):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Job Title', job_title])
    writer.writerow([])
    writer.writerow(['Board ID', 'Piece ID', 'Length', 'Width', 'Rotated', 'X', 'Y'])
    for board in purchased_boards:
        for cut in board['cuts']:
            writer.writerow([
                board['board_id'],
                cut['piece']['id'],
                f"{cut['length']:.3f}",
                f"{cut['width']:.3f}",
                cut['rotated'],
                round(cut['x'], 2),
                round(cut['y'], 2)
            ])
    output.seek(0)
    return output.getvalue()

def save_plan_to_json(plan, leftovers, required_df, job_title):
    data = {
        'job_title': job_title,
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
        pd.DataFrame(data.get('required_input', [])),
        data.get('job_title', '')
    )

def save_plan_to_yaml(plan, leftovers, required_df, job_title):
    data = {
        'job_title': job_title,
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
        pd.DataFrame(data.get('required_input', [])),
        data.get('job_title', '')
    )

# ---- Streamlit App ----
st.set_page_config(page_title="Lumber Purchase Optimizer", layout="wide")
st.title("\U0001F4D0 Lumber Purchase Optimizer")

job_title = st.text_input("\U0001F4DD Job Title", value=st.session_state.get("job_title", "Untitled Job"))
st.session_state["job_title"] = job_title

st.sidebar.header("Cut & Cost Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2, "Project Name": ""}])

required_df = st.data_editor(
    st.session_state.get('required_df', default_cut_df()),
    num_rows="dynamic", use_container_width=True, column_config={"Project Name": st.column_config.TextColumn("Project Name", required=False)}, key="required_df_editor")

required_df = required_df.sort_values(by="Project Name")

if st.button("\U0001FAA8 Optimize Lumber Purchase"):
    purchase_plan, leftovers, total_cost = optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers
    st.session_state.required_df = required_df

    total_board_feet = sum(b['board_feet'] for b in purchase_plan)
    st.success(f"Optimization complete! Total board feet purchased: {total_board_feet:.2f}, Estimated Total Cost: ${total_cost:.2f}")

    csv_data = generate_csv(purchase_plan, job_title=job_title)
    pdf_data = generate_pdf(purchase_plan, leftovers, job_title=job_title)

    st.download_button("\U0001F4C4 Download CSV", csv_data, file_name="purchase_plan.csv", mime="text/csv")
    st.download_button("\U0001F4C4 Download PDF", pdf_data, file_name="purchase_plan.pdf")

    if leftovers:
        st.warning("Some required pieces could not be allocated to any board. Please review the leftover pieces.")

st.sidebar.header("Plan Persistence Options")
file_format = st.sidebar.radio("Select File Format", options=["JSON", "YAML"])
save_plan_button = st.sidebar.button("Save Plan")
load_file = st.sidebar.file_uploader("Load Plan File", type=["json", "yaml", "yml"])

if save_plan_button and 'purchase_plan' in st.session_state:
    if file_format == "JSON":
        saved_data = save_plan_to_json(
            st.session_state.purchase_plan,
            st.session_state.leftovers,
            st.session_state.required_df,
            job_title
        )
        st.sidebar.download_button("Download JSON", saved_data, file_name="purchase_plan.json", mime="application/json")
    else:
        saved_data = save_plan_to_yaml(
            st.session_state.purchase_plan,
            st.session_state.leftovers,
            st.session_state.required_df,
            job_title
        )
        st.sidebar.download_button("Download YAML", saved_data, file_name="purchase_plan.yaml", mime="text/yaml")

if load_file:
    file_content = load_file.read().decode("utf-8")
    if load_file.name.endswith(".json"):
        purchase_plan, leftovers, required_df, job_title_loaded = load_plan_from_json(file_content)
    else:
        purchase_plan, leftovers, required_df, job_title_loaded = load_plan_from_yaml(file_content)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers
    st.session_state.required_df = required_df
    st.session_state.job_title = job_title_loaded
    st.rerun()

