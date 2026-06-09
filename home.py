import streamlit as st

from ui.theme import page_header

page_header(
    "VN AIDEOM-VN",
    "AI-Driven Decision Optimization Model for Vietnam",
)

st.markdown(
    """
    Hệ thống web tích hợp **12 bài toán mô hình ra quyết định**,
    hỗ trợ phân tích chính sách phát triển kinh tế Việt Nam
    trong kỷ nguyên trí tuệ nhân tạo.
    """
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Bài toán", "12")
col2.metric("Nhóm mô hình", "4")
col3.metric("Kịch bản Bài 12", "5")
col4.metric("Tác nhân AI", "12")

st.divider()
st.subheader("📚 12 bài toán theo 4 cấp độ")

with st.expander("🟢 Cấp độ dễ — Làm quen mô hình", expanded=True):
    st.markdown(
        """
        - **Bài 1:** Cobb-Douglas mở rộng và dự báo GDP.
        - **Bài 2:** LP phân bổ ngân sách số.
        - **Bài 3:** Chỉ số ưu tiên 10 ngành.
        """
    )

with st.expander("🟡 Cấp độ trung bình — Tối ưu cổ điển"):
    st.markdown(
        """
        - **Bài 4:** Phân bổ vùng và công bằng số.
        - **Bài 5:** Lựa chọn danh mục 15 dự án.
        - **Bài 6:** TOPSIS, Entropy và AHP.
        """
    )

with st.expander("🟠 Cấp độ nâng cao — Đa mục tiêu và động"):
    st.markdown(
        """
        - **Bài 7:** NSGA-II và đường biên Pareto.
        - **Bài 8:** Tối ưu liên thời gian 2026-2035.
        - **Bài 9:** Lao động và trí tuệ nhân tạo.
        """
    )

with st.expander("🔴 Cấp độ chuyên sâu — Bất định và AI"):
    st.markdown(
        """
        - **Bài 10:** Stochastic Programming.
        - **Bài 11:** Q-learning và DQN.
        - **Bài 12:** Hệ thống AIDEOM-VN tích hợp.
        """
    )

st.divider()
st.subheader("🧭 Quy trình sử dụng")
st.markdown(
    """
    1. Chọn bài toán ở thanh menu bên trái.
    2. Điều chỉnh tham số đầu vào.
    3. Bấm **Chạy mô hình**.
    4. Xem bảng, KPI và biểu đồ.
    5. Bấm **Phân tích kết quả bằng AI**.
    6. Tải kết quả để đưa vào báo cáo.
    """
)

st.caption(
    "Đây là khung giao diện. Thuật toán từng bài sẽ được tích hợp ở các bước tiếp theo."
)
