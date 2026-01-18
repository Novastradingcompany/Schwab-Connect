# ============================
# core/export_utils.py
# ============================

import pandas as pd
import base64
from io import BytesIO
from datetime import datetime


# =========================================================
# 💾 CSV Export
# =========================================================
def download_csv_button(df: pd.DataFrame, filename_prefix="NovaScan"):
    """Generate a Streamlit-friendly CSV download link."""
    if df is None or df.empty:
        return None

    now_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{filename_prefix}_{now_str}.csv"
    csv_bytes = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">💾 Download CSV</a>'
    return href


# =========================================================
# 📊 Excel Export
# =========================================================
def download_excel_button(df: pd.DataFrame, filename_prefix="NovaScan"):
    """Generate a Streamlit-friendly Excel (.xlsx) download link with formatting."""
    if df is None or df.empty:
        return None

    now_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{filename_prefix}_{now_str}.xlsx"

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="NovaScan")

        # Basic formatting
        sheet = writer.sheets["NovaScan"]
        for column_cells in sheet.columns:
            max_length = 0
            col = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            adjusted_width = (max_length + 2)
            sheet.column_dimensions[col].width = adjusted_width

    excel_bytes = output.getvalue()
    b64 = base64.b64encode(excel_bytes).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📊 Download Excel</a>'
    return href
