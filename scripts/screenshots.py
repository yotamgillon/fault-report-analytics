"""Capture the dashboard screenshots used in the README (a dev tool, not part of `make all`).

    make screenshots        # installs the Chromium that Playwright drives, then runs this

It starts the Streamlit app on a free local port, takes three screenshots of the real running
dashboard and writes them to docs/img/. The data is the synthetic dataset built by `make all`,
so run that first. The Excel screenshots are taken by hand (see docs/img/CAPTURE.md).
"""

import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
CLEAN_CSV = ROOT / "output" / "clean_reports.csv"
WIDTH = 1440


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_server(port: int, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("the Streamlit app did not start in time")


def open_dashboard(page: Page, port: int) -> None:
    page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
    page.wait_for_selector("text=Vendor scorecard", timeout=90_000)
    page.wait_for_selector(".js-plotly-plot", timeout=90_000)
    page.wait_for_timeout(2_500)  # let the charts finish animating


def only_select(page: Page, which: int, value: str) -> None:
    """Clear a sidebar multiselect (0 = site, 1 = vendor, 2 = severity) and pick one value."""
    box = page.locator('[data-testid="stSidebar"] [data-testid="stMultiSelect"]').nth(which)
    box.locator('[aria-label="Clear all"]').click()
    box.get_by_role("combobox").fill(value)
    page.keyboard.press("Enter")
    page.keyboard.press("Escape")
    page.wait_for_timeout(2_500)  # Streamlit reruns the script


def shoot_page(page: Page, name: str, height: int | None = None) -> None:
    """Screenshot the whole app (sidebar + page); ``height`` crops to the top part."""
    content = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    bottom = int(content["y"] + content["height"]) + 40
    page.screenshot(
        path=OUT / name, clip={"x": 0, "y": 0, "width": WIDTH, "height": height or bottom}
    )
    print(f"wrote docs/img/{name}")


def main() -> int:
    if not CLEAN_CSV.exists():
        print("output/clean_reports.csv is missing: run `make all` first.", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py", "--server.headless=true",
         f"--server.port={port}", "--browser.gatherUsageStats=false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        wait_for_server(port)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": WIDTH, "height": 3000})
            open_dashboard(page, port)
            shoot_page(page, "dashboard_overview.png")

            page.get_by_role("tab", name="SLA (demo values)").click()
            page.wait_for_timeout(1_000)
            tabs = page.locator('[data-testid="stTabs"]').bounding_box()
            page.screenshot(
                path=OUT / "dashboard_sla_demo.png",
                clip={"x": tabs["x"] - 20, "y": tabs["y"] - 60, "width": tabs["width"] + 40,
                      "height": tabs["height"] + 80},
            )  # fmt: skip
            print("wrote docs/img/dashboard_sla_demo.png")

            open_dashboard(page, port)  # fresh session, then a filter that makes groups small
            only_select(page, 2, "Critical")
            shoot_page(page, "dashboard_filtered_low_sample.png", height=1250)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=15)
    return 0


if __name__ == "__main__":
    sys.exit(main())
