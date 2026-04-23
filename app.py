import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress
from scipy.signal import savgol_filter
import io
import zipfile
import os

# ページ設定
st.set_page_config(page_title="Bucci Plot (TSC)解析システム", layout="wide")
st.title("Bucci Plot (TSC)解析システム")

# --- サイドバー：解析設定 ---
st.sidebar.header("1. データの前処理")
enable_smoothing = st.sidebar.checkbox("スムージングを有効にする", value=True)
smooth_window = st.sidebar.slider("スムージング強度 (窓幅)", 5, 51, 11, step=2)

st.sidebar.header("2. ベースライン設定")
poly_degree = st.sidebar.slider("ベースラインの多項式次数", 1, 5, 1)

st.sidebar.header("3. 解析範囲の設定 [℃]")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("測定データ (CSV: 時間[min], 温度[℃], 電流[A]) を選択", type=["csv"])

if uploaded_file is not None:
    # 元のファイル名から拡張子を除いた部分を取得
    raw_filename = os.path.splitext(uploaded_file.name)[0]

    try:
        # データ読み込み
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='shift-jis')

        t_min = df.iloc[:, 0].values
        T_degC = df.iloc[:, 1].values
        I_raw = df.iloc[:, 2].values
        
        t_sec = t_min * 60.0
        T_K = T_degC + 273.15
        
        # スライダーの動的範囲設定
        min_t_data = float(np.min(T_degC))
        max_t_data = float(np.max(T_degC))
        
        base_low = st.sidebar.slider("ベースライン: 低温側範囲", min_t_data, max_t_data, (min_t_data, min_t_data + (max_t_data-min_t_data)*0.1))
        base_high = st.sidebar.slider("ベースライン: 高温側範囲", min_t_data, max_t_data, (max_t_data - (max_t_data-min_t_data)*0.1, max_t_data))
        
        peak_range = st.sidebar.slider("★単一ピーク抽出範囲 (積分&β算出範囲)", min_t_data, max_t_data, (min_t_data + (max_t_data-min_t_data)*0.2, max_t_data - (max_t_data-min_t_data)*0.2))
        
        bucci_range = st.sidebar.slider("Bucci近似: 直線フィッティング範囲", peak_range[0], peak_range[1], (peak_range[0], (peak_range[0]+peak_range[1])/2))

        # --- 計算 ---
        # 1. スムージング
        I_target = I_raw
        if enable_smoothing:
            w = smooth_window if smooth_window < len(I_raw) else len(I_raw)//2 * 2 - 1
            I_target = savgol_filter(I_raw, w, 2)

        # 2. ベースライン除去
        mask_base = ((T_degC >= base_low[0]) & (T_degC <= base_low[1])) | ((T_degC >= base_high[0]) & (T_degC <= base_high[1]))
        coef_base = np.polyfit(T_degC[mask_base], I_target[mask_base], poly_degree)
        I_baseline = np.polyval(coef_base, T_degC)
        I_clean = I_target - I_baseline

        # 3. 指定されたピーク範囲内での計算（積分 & 昇温速度）
        mask_peak = (T_degC >= peak_range[0]) & (T_degC <= peak_range[1])
        T_K_peak = T_K[mask_peak]
        t_sec_peak = t_sec[mask_peak]
        I_clean_peak = I_clean[mask_peak]
        
        slope_beta, _, r_v, _, _ = linregress(t_sec_peak, T_K_peak)
        beta = slope_beta
        
        I_clean_safe = np.where(I_clean_peak <= 0, 1e-25, I_clean_peak)
        T_rev = T_K_peak[::-1]
        I_rev = I_clean_safe[::-1]
        integral_rev = np.abs(cumulative_trapezoid(I_rev, T_rev, initial=0))
        P_T = integral_rev[::-1] / beta 
        tau = P_T / I_clean_safe
        x_bucci_peak = 1000.0 / T_K_peak
        y_bucci_peak = np.log(tau)

        # 4. 直線近似 (Bucci近似範囲内)
        mask_fit = (T_degC[mask_peak] >= bucci_range[0]) & (T_degC[mask_peak] <= bucci_range[1])
        x_fit = x_bucci_peak[mask_fit]
        y_fit = y_bucci_peak[mask_fit]
        
        if len(x_fit) > 1:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            Ea = slope * 8.617e-5 * 1000
            tau0 = np.exp(intercept)
        else:
            st.error("直線近似範囲内のデータ点が不足しています。")
            st.stop()

        # --- 結果表示 ---
        st.subheader("解析結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("昇温速度 β (範囲内)", f"{beta*60:.2f} K/min")
        c2.metric("昇温直線性 R²", f"{r_v**2:.4f}")
        c3.metric("活性化エネルギー Ea", f"{Ea:.3f} eV")
        c4.metric("頻度因子 τ₀", f"{tau0:.2e} s")

        # --- グラフ表示 ---
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.plot(T_degC, I_raw, 'k:', alpha=0.3, label='Raw Data')
            ax1.plot(T_degC, I_baseline, 'r--', label='Baseline')
            ax1.plot(T_degC, I_clean, 'b-', label='Cleaned')
            ax1.axvspan(base_low[0], base_low[1], color='red', alpha=0.1, label='Base Fit')
            ax1.axvspan(base_high[0], base_high[1], color='red', alpha=0.1)
            ax1.axvspan(peak_range[0], peak_range[1], color='green', alpha=0.08, label='Peak Range')
            ax1.set_xlabel("Temperature [°C]"); ax1.set_ylabel("Current [A]")
            ax1.legend(); ax1.grid(True)
            st.pyplot(fig1)

        with col_g2:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.scatter(x_bucci_peak, y_bucci_peak, color='gray', s=5, alpha=0.4)
            ax2.scatter(x_fit, y_fit, color='blue', s=15, label='Fit Range')
            x_line = np.linspace(min(x_fit)*0.98, max(x_fit)*1.02, 50)
            ax2.plot(x_line, np.polyval([slope, intercept], x_line), 'r-', label=f'Ea={Ea:.3f}eV')
            ax2.set_xlabel("1000 / T [1/K]"); ax2.set_ylabel("ln(tau)")
            ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

        # --- 保存データの作成 ---
        summary_df = pd.DataFrame({
            "Parameter": ["Activation Energy (Ea)", "Pre-exponential factor (tau0)", "Heating Rate (beta)", "Heating Linearity (R2)", "Baseline Low", "Baseline High", "Peak Range", "Bucci Range"],
            "Value": [f"{Ea:.4f}", f"{tau0:.4e}", f"{beta*60:.4f}", f"{r_v**2:.5f}", f"{base_low}", f"{base_high}", f"{peak_range}", f"{bucci_range}"],
            "Unit": ["eV", "s", "K/min", "-", "degC", "degC", "degC", "degC"]
        })
        buf1 = io.BytesIO(); fig1.savefig(buf1, format="png", dpi=300, bbox_inches='tight')
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
        tsc_df = pd.DataFrame({"Temp_degC": T_degC, "I_raw": I_raw, "I_baseline": I_baseline, "I_clean": I_clean})
        bucci_df = pd.DataFrame({"1000/T": x_bucci_peak, "ln_tau": y_bucci_peak})

        # --- ダウンロードボタンの表示 ---
        st.divider()
        st.subheader("データの保存")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.download_button(f"🖼️ TSCグラフ保存", buf1.getvalue(), f"{raw_filename}_tsc_plot.png", "image/png")
        with col_img2:
            st.download_button(f"🖼️ Bucciグラフ保存", buf2.getvalue(), f"{raw_filename}_bucci_plot.png", "image/png")

        col_csv1, col_csv2, col_csv3 = st.columns(3)
        with col_csv1:
            st.download_button("📄 TSCデータ (CSV)", tsc_df.to_csv(index=False).encode('utf-8'), f"{raw_filename}_processed_tsc.csv", "text/csv")
        with col_csv2:
            st.download_button("📄 Bucciプロットデータ (CSV)", bucci_df.to_csv(index=False).encode('utf-8'), f"{raw_filename}_bucci_data.csv", "text/csv")
        with col_csv3:
            st.download_button("📊 解析サマリー (CSV)", summary_df.to_csv(index=False).encode('utf-8'), f"{raw_filename}_summary_results.csv", "text/csv")

        # 一括ダウンロード（Zip）
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            zip_file.writestr(f"{raw_filename}_summary_results.csv", summary_df.to_csv(index=False))
            zip_file.writestr(f"{raw_filename}_processed_tsc.csv", tsc_df.to_csv(index=False))
            zip_file.writestr(f"{raw_filename}_bucci_data.csv", bucci_df.to_csv(index=False))
            zip_file.writestr(f"{raw_filename}_tsc_plot.png", buf1.getvalue())
            zip_file.writestr(f"{raw_filename}_bucci_plot.png", buf2.getvalue())
        
        st.download_button(
            label="📁 全ての解析結果を一括保存 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{raw_filename}_bucci_analysis_results.zip",
            mime="application/zip",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"解析エラー: {e}")