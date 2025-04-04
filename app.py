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
    # length (inches) and width (inches) conversion: board feet = (thickness * width * length) / 144
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
        if length is None or width is None:
            continue
        for _ in range(quantity):
            pieces.append({
                'length': length,
                'width': width,
                'id': f"{length:.3f}x{width:.3f}"
            })
    # Sort pieces largest-first (by max dimension)
    return sorted(pieces, key=lambda x: max(x['length'], x['width']), reverse=True)

# --- Packing Algorithm (Free-Rectangle Based) ---
def try_place_pieces(board, pieces, kerf):
    # board: dict with 'length' and 'width' in inches.
    free_rectangles = [{'x': 0, 'y': 0, 'length': board['length'], 'width': board['width']}]
    placements = []
    remaining = []

    for piece in pieces:
        placed = False
        for rect in free_rectangles.copy():
            for rotated in [False, True]:
                # Add kerf to piece dimensions
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
                    # Create new free rectangles from the remaining space
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

# --- New Optimization: Determine Purchased Lumber ---
def optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf):
    pieces = generate_required_pieces(required_df)
    purchased_boards = []
    board_counter = 1
    
    # Allowed board lengths in feet and corresponding inches
    allowed_lengths_ft = [5, 6, 8, 10, 12]
    allowed_lengths_in = [ft * 12 for ft in allowed_lengths_ft]
    # Allowed board widths (inches): 4" up to 12" in 1/2" increments.
    allowed_widths = [4 + 0.5*i for i in range(int((12 - 4) / 0.5) + 1)]
    
    # Build list of allowed board candidates.
    allowed_boards = []
    for L in allowed_lengths_in:
        for W in allowed_widths:
            allowed_boards.append({
                'length': L,      # in inches
                'width': W,       # in inches
                'length_ft': L/12, # in feet (for cost display)
                'width_in': W     # in inches
            })
    
    # Greedy algorithm: while there are pieces to cut, choose the board candidate that gives highest utilization.
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
        
        # Remove placed pieces from list
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

# ---- PDF Generation (Re-used from original) ----
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

def generate_pdf(purchased_boards, leftovers=None):
    buffer = io.BytesIO()
    page_width, page_height = 8.5, 11  # Letter size
    drawing_title_font = 12
    drawing_axis_font = 10
    drawing_piece_font = 8
    drawing_linewidth = 1.0

    with PdfPages(buffer) as pdf:
        # Generate a page for each purchased board
        for board in purchased_boards:
            b = board['board']
            board_title = (
                f"Board {board['board_id']} - "
                f"{b['length_ft']:.1f} ft x {b['width_in']:.1f}\" "
                f"({board['board_feet']:.2f} bf, Utilization: {board['utilization']*100:.1f}%)"
            )
            fig = plt.figure(figsize=(page_width, page_height))
            gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.3)
            
            ax_draw = fig.add_subplot(gs[0])
            ax_draw.set_title(board_title, fontsize=drawing_title_font, color='red')
            ax_draw.set_xlabel("Inches", fontsize=drawing_axis_font)
            ax_draw.set_ylabel("Inches", fontsize=drawing_axis_font)
            ax_draw.set_xlim(0, b['length'])
            ax_draw.set_ylim(0, b['width'])
            ax_draw.set_aspect('equal', adjustable='box')
            
            for cut in board['cuts']:
                rect = patches.Rectangle(
                    (cut['x'], cut['y']),
                    cut['length'],
                    cut['width'],
                    linewidth=drawing_linewidth,
                    edgecolor='black',
                    facecolor='lightgrey'
                )
                ax_draw.add_patch(rect)
                piece_label = (
                    f"{to_fraction_string(cut['piece']['length'])}\" x "
                    f"{to_fraction_string(cut['piece']['width'])}\""
                )
                ax_draw.text(
                    cut['x'] + cut['length'] / 2,
                    cut['y'] + cut['width'] / 2,
                    piece_label,
                    ha='center', va='center',
                    fontsize=drawing_piece_font,
                    color='red'
                )
            
            ax_text = fig.add_subplot(gs[1])
            ax_text.axis('off')
            text_lines = []
            header = f"{'Piece ID':<12} {'X':>6} {'Y':>6} {'L':>8} {'W':>8} {'Rotated':>8}"
            text_lines.append(header)
            text_lines.append("-" * len(header))
            for cut in board['cuts']:
                piece_id = cut['piece']['id']
                x_str = to_fraction_string(cut['x'])
                y_str = to_fraction_string(cut['y'])
                length_str = to_fraction_string(cut['length'])
                width_str = to_fraction_string(cut['width'])
                rotated_str = "Yes" if cut.get('rotated', False) else "No"
                line = f"{piece_id:<12} {x_str:>6} {y_str:>6} {length_str:>8} {width_str:>8} {rotated_str:>8}"
                text_lines.append(line)
            text_block = "\n".join(text_lines)
            ax_text.text(0, 1, text_block, fontsize=10, family='monospace', va='top')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Add an extra page for leftover pieces if any exist.
        if leftovers:
            fig, ax = plt.subplots(figsize=(page_width, page_height))
            ax.axis('off')
            ax.set_title("Leftover Pieces", fontsize=12)
            text_lines = []
            header = f"{'Length':>8} {'Width':>8}"
            text_lines.append(header)
            text_lines.append("-" * len(header))
            for piece in leftovers:
                length_str = to_fraction_string(piece['length'])
                width_str = to_fraction_string(piece['width'])
                line = f"{length_str:>8} {width_str:>8}"
                text_lines.append(line)
            text_block = "\n".join(text_lines)
            ax.text(0, 1, text_block, fontsize=10, family='monospace', va='top')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # ---- Summary Page: Boards Needed ----
        # Aggregate the purchased boards by their dimensions
        summary_data = {}
        for board in purchased_boards:
            dims = (board['board']['length_ft'], board['board']['width_in'])
            summary_data[dims] = summary_data.get(dims, 0) + 1

        # Create summary page
        fig, ax = plt.subplots(figsize=(page_width, page_height))
        ax.axis('off')
        ax.set_title("Summary of Boards Purchased", fontsize=14)
        text_lines = []
        text_lines.append(f"{'Board Dimensions':<20} {'Quantity':>10}")
        text_lines.append("-" * 32)
        for dims, qty in sorted(summary_data.items()):
            length_ft, width_in = dims
            # Format: "X ft x Y" where Y is in inches.
            dims_str = f"{length_ft:.1f} ft x {width_in:.1f}\""
            line = f"{dims_str:<20} {qty:>10}"
            text_lines.append(line)
        text = "\n".join(text_lines)
        ax.text(0.1, 0.9, text, fontsize=12, family='monospace', va='top')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    buffer.seek(0)
    return buffer

def generate_csv(purchased_boards):
    output = io.StringIO()
    writer = csv.writer(output)
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

# ---- Plan Persistence Functions (JSON/YAML) ----
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

# Sidebar: Settings and Persistence Options
st.sidebar.header("Cut & Cost Settings")
kerf = st.sidebar.number_input("Kerf Size (inches)", value=0.125, step=0.001, format="%.3f")
thickness = st.sidebar.number_input("Board Thickness (inches)", value=0.75, step=0.01)
cost_per_bf = st.sidebar.number_input("Cost per Board Foot ($)", value=5.00, step=0.01)

st.sidebar.header("Plan Persistence Options")
file_format = st.sidebar.radio("Select File Format", options=["JSON", "YAML"])
save_plan_button = st.sidebar.button("Save Plan")
load_file = st.sidebar.file_uploader("Load Plan File", type=["json", "yaml", "yml"])

# Main UI: Required Cuts
st.subheader("Required Cuts")
def default_cut_df():
    return pd.DataFrame([{"Length": "24", "Width": "6", "Quantity": 2}])
required_df = st.data_editor(
    st.session_state.get('required_df', default_cut_df()),
    num_rows="dynamic", use_container_width=True)

if st.button("🔨 Optimize Lumber Purchase"):
    purchase_plan, leftovers, total_cost = optimize_lumber_purchase(required_df, kerf, thickness, cost_per_bf)
    st.session_state.purchase_plan = purchase_plan
    st.session_state.leftovers = leftovers
    st.session_state.required_df = required_df

    total_board_feet = sum(b['board_feet'] for b in purchase_plan)
    st.success(f"Optimization complete! Total board feet purchased: {total_board_feet:.2f}, Estimated Total Cost: ${total_cost:.2f}")

    csv_data = generate_csv(purchase_plan)
    pdf_data = generate_pdf(purchase_plan, leftovers)

    st.download_button("📄 Download CSV", csv_data, file_name="purchase_plan.csv", mime="text/csv")
    st.download_button("📄 Download PDF", pdf_data, file_name="purchase_plan.pdf")
    if leftovers:
        st.warning("Some required pieces could not be allocated to any board. Please review the leftover pieces.")

# Sidebar: Save/Load Plan Actions
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
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    else:
        st.write("Please manually refresh the page to see the loaded plan.")
