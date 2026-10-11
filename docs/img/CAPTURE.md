# Excel screenshots to capture

The dashboard screenshots are produced by `make screenshots`. The six Excel screenshots below are
taken by hand in desktop Excel. Each file in this folder with an `excel_` prefix is currently a
grey placeholder: save your capture over it with the **same file name** and the README picks it up.

Open `output/field_ops_report.xlsx` (build it with `make all`). The first time, click **Enable
Editing** if Excel opens it in Protected View and wait for the status bar to stop saying
"Calculate": the workbook computes every formula when it opens.

| File | Sheet | What to show |
|---|---|---|
| `excel_dashboard.png` | Dashboard | Default selection (All vendors / All sites). Scroll to the top so the title, both yellow dropdowns, the eight KPI cards and all four charts are visible. |
| `excel_dashboard_filtered.png` | Dashboard | Set the Vendor dropdown to **Vendor F** and the Site dropdown to **Ridgeview**. The KPI cards must show different numbers than the default capture. |
| `excel_by_vendor.png` | By_Vendor | The whole table including the total row: colour scale on the median column, red SLA-breach cells, tier and SLA target pulled in by XLOOKUP. |
| `excel_by_system.png` | By_System | The first ~15 rows: repeat rate, fleet repeat rate, the "vs fleet" columns and the amber highlights. |
| `excel_formulas.png` | Data | Click the first cell under `vendor_tier` so the formula bar shows the XLOOKUP. Keep the formula bar, column headers and about ten rows in view. |
| `excel_backlog.png` | Backlog | The overall row plus the by-site and by-vendor blocks (the blue heat shading). |

Tips
- Zoom 100%, window about 1400 px wide; capture the sheet area only.
- **Crop out** the title bar, any account name or profile picture, and the file path in the window
  title: only the workbook content should be in the image.
- PNG, roughly 1400 px wide. Larger is fine but keep each file under about 500 KB.
- If Excel shows a "repair" prompt, a `#NAME?` or any other error, do not capture; report it.
