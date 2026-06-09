from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from ui.theme import apply_theme
except Exception:
    def apply_theme() -> None:
        st.markdown(
            """
            <style>
            .stApp {
                background: #FFF9FB;
                color: #49313D;
            }

            section[data-testid="stSidebar"] {
                background: #FCEEF4;
            }

            h1, h2, h3 {
                color: #49313D !important;
            }

            .stButton > button {
                background: #D989A5 !important;
                color: white !important;
                border-radius: 12px !important;
                border: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def missing_page_factory(title: str, rel_path: str):
    def _missing_page():
        apply_theme()

        st.title(title)

        st.error(
            "Trang này chưa chạy được vì file page tương ứng chưa được upload lên GitHub."
        )

        st.markdown("File cần có:")

        st.code(rel_path)

        st.info(
            "Hãy upload lại thư mục `pages`, `core`, `services`, `ui`, `data` "
            "từ máy lên GitHub rồi reboot Streamlit Cloud."
        )

    return _missing_page


def make_page(
    rel_path: str,
    title: str,
    icon: str,
    default: bool = False,
):
    page_file = ROOT / rel_path

    if page_file.exists():
        return st.Page(
            rel_path,
            title=title,
            icon=icon,
            default=default,
        )

    return st.Page(
        missing_page_factory(title, rel_path),
        title=title,
        icon="⚠️",
        default=default,
    )


apply_theme()


pages = [
    make_page(
        "pages/home.py",
        "Trang chủ",
        "🏠",
        default=True,
    ),
    make_page(
        "pages/bai01.py",
        "Bài 1 — Cobb-Douglas + AI",
        "🌱",
    ),
    make_page(
        "pages/bai02.py",
        "Bài 2 — LP ngân sách số",
        "💰",
    ),
    make_page(
        "pages/bai03.py",
        "Bài 3 — Priority 10 ngành",
        "📊",
    ),
    make_page(
        "pages/bai04.py",
        "Bài 4 — LP ngành-vùng",
        "🗺️",
    ),
    make_page(
        "pages/bai05.py",
        "Bài 5 — MIP 15 dự án",
        "🎯",
    ),
    make_page(
        "pages/bai06.py",
        "Bài 6 — TOPSIS 6 vùng",
        "🏆",
    ),
    make_page(
        "pages/bai07.py",
        "Bài 7 — NSGA-II Pareto",
        "🌐",
    ),
    make_page(
        "pages/bai08.py",
        "Bài 8 — Tối ưu động",
        "⏳",
    ),
    make_page(
        "pages/bai09.py",
        "Bài 9 — Lao động & AI",
        "👥",
    ),
    make_page(
        "pages/bai10.py",
        "Bài 10 — Stochastic SP",
        "🎲",
    ),
    make_page(
        "pages/bai11.py",
        "Bài 11 — Q-learning RL",
        "☯️",
    ),
    make_page(
        "pages/bai12.py",
        "Bài 12 — AIDEOM tích hợp",
        "VN",
    ),
]


navigation = st.navigation(
    pages,
    position="sidebar",
)

navigation.run()
