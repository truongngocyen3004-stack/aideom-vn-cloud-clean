from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import streamlit as st

from ui.theme import apply_theme

st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

pages = [
    st.Page(
        "pages/home.py",
        title="Trang chủ",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "pages/bai01.py",
        title="Bài 1 — Cobb-Douglas + AI",
        icon="🌱",
    ),
    st.Page(
        "pages/bai02.py",
        title="Bài 2 — LP ngân sách số",
        icon="💰",
    ),
    st.Page(
        "pages/bai03.py",
        title="Bài 3 — Priority 10 ngành",
        icon="📊",
    ),
    st.Page(
        "pages/bai04.py",
        title="Bài 4 — LP ngành-vùng",
        icon="🗺️",
    ),
    st.Page(
        "pages/bai05.py",
        title="Bài 5 — MIP 15 dự án",
        icon="🎯",
    ),
    st.Page(
        "pages/bai06.py",
        title="Bài 6 — TOPSIS 6 vùng",
        icon="🏆",
    ),
    st.Page(
        "pages/bai07.py",
        title="Bài 7 — NSGA-II Pareto",
        icon="🌐",
    ),
    st.Page(
        "pages/bai08.py",
        title="Bài 8 — Tối ưu động",
        icon="⌛",
    ),
    st.Page(
        "pages/bai09.py",
        title="Bài 9 — Lao động & AI",
        icon="👥",
    ),
    st.Page(
        "pages/bai10.py",
        title="Bài 10 — Stochastic SP",
        icon="🎲",
    ),
    st.Page(
        "pages/bai11.py",
        title="Bài 11 — Q-learning RL",
        icon="🤖",
    ),
    st.Page(
        "pages/bai12.py",
        title="Bài 12 — AIDEOM tích hợp",
        icon="🇻🇳",
    ),
]

navigation = st.navigation(
    pages,
    position="sidebar",
)

navigation.run()
