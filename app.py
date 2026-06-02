"""
[목적] Streamlit 기반 구강 건강 위험도 예측 웹 애플리케이션 (메인)
[실행] streamlit run app.py
[탭 구성]
  1. 스마트폰 과의존 — 개념·진단기준·S-Scale 안내
  2. 구강 예측하기  — S-Scale 자가진단 + XGBoost 위험도 예측
  3. 맞춤형 솔루션  — 4가지 유형별 맞춤 행동 가이드
  4. 자료실         — 통계 예측지도·관리 가이드·프로그램 안내
"""

import io
import json
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="Cheer up(치아 업) Clinic",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. 전역 변수 및 통합 경로 설정
# ==========================================
DATA_PATH = Path("./data/processed/kyrbs2020_clean_v1.csv")
MODELS_DIR = Path("./models")

APP_TITLE = "Cheer up(치아 업)Clinic"
APP_SUBTITLE = "청소년 스마트폰 의존과 수면의 질에 따른 구강 건강 위험도 예측 모델 개발 및 개인 맞춤형 솔루션"
TIME_OPTIONS = ["3시간 이하", "3~5시간", "5~8시간", "8시간 이상"]

SMARTPHONE_QUESTIONS = [
    "스마트폰 이용시간을 조절하는 것이 어렵다.",
    "스마트폰 이용시간을 줄이려고 시도해보았지만 실패한다.",
    "스마트폰 이용시간을 조절하는 것이 내 마음대로 안 된다.",
    "스마트폰 이용시간이 점점 늘어난다.",
    "스마트폰이 옆에 없으면 안절부절못하고 초조해진다.",
    "스마트폰이 생각나서 다른 일에 집중하기 힘들다.",
    "스마트폰이 없으면 일상생활이 힘들 것 같다는 생각이 든다.",
    "스마트폰 이용 때문에 건강에 문제가 생긴 적이 있다.",
    "스마트폰 이용 때문에 가족(친구)과 다툰 적이 있다.",
    "스마트폰 이용 때문에 해야 할 일(공부 등)에 지장을 받는다."
]

# ==========================================
# 3. CSS 스타일링
# ==========================================
CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none !important;}
    .block-container { padding-top: 0.8rem; padding-bottom: 2.5rem; max-width: 1480px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 0.85rem; width: 100%; margin-top: 0.7rem; margin-bottom: 1.15rem; }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0; min-height: 92px; font-size: 1.32rem; font-weight: 800;
        border-radius: 20px; border: 1px solid #ddd6c8; background: #f7f2e8;
        color: #403830; justify-content: center; box-shadow: 0 10px 24px rgba(58, 41, 20, 0.04);
        padding: 0.85rem 1.1rem;
    }
    .stTabs [data-baseweb="tab"] p { font-size: 1.32rem !important; font-weight: 800 !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #fffaf4 0%, #efe3d2 100%) !important;
        border: 1.5px solid #c9b291 !important; color: #9b5450 !important;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #fbf7ef 0%, #f5eee3 45%, #efe2d1 100%);
        border: 1px solid #ddd2bf; border-radius: 32px; padding: 38px 42px 34px 42px;
        box-shadow: 0 18px 36px rgba(52, 33, 14, 0.07); margin-bottom: 1.1rem;
    }
    .hero-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 1.35rem; align-items: stretch; }
    .hero-title { font-size: 3.45rem; line-height: 1.04; font-weight: 900; letter-spacing: -0.04em; color: #b55a5b; margin: 0 0 0.7rem 0; }
    .hero-subtitle { font-size: 1.17rem; line-height: 1.75; color: #4a4039; margin: 0; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 0.65rem; margin-top: 1.15rem; }
    .badge {
        display: inline-flex; align-items: center; padding: 0.58rem 0.95rem; border-radius: 999px;
        background: rgba(255,255,255,0.74); border: 1px solid #eadfd0; font-weight: 700; color: #5c4b43; font-size: 0.96rem;
    }
    .hero-panel { background: rgba(255,255,255,0.72); border: 1px solid #e8ddcd; border-radius: 24px; padding: 1.1rem 1.1rem 1rem 1.1rem; }
    .hero-panel-title { font-size: 1.15rem; font-weight: 800; color: #69584d; margin-bottom: 0.7rem; }
    .mini-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
    .mini-stat { border-radius: 18px; background: #fffdf8; border: 1px solid #ede2d3; padding: 1rem; }
    .mini-stat-label { color: #7a6b61; font-size: 0.9rem; margin-bottom: 0.18rem; }
    .mini-stat-value { font-size: 1.42rem; font-weight: 900; color: #312a26; }
    
    .section-title { margin: 0.1rem 0 0.8rem 0; font-size: 1.42rem; font-weight: 900; color: #2d2724; }
    .section-desc { color: #72665d; margin-top: -0.2rem; margin-bottom: 1rem; line-height: 1.7; }
    .card { border-radius: 22px; border: 1px solid #e8dfd2; background: #fffdf9; padding: 1.2rem; box-shadow: 0 12px 28px rgba(60, 42, 23, 0.05); height: 100%; }
    .kpi-card { border-radius: 20px; border: 1px solid #eee4d6; background: linear-gradient(180deg, #fffdf9 0%, #faf5ed 100%); padding: 1rem 1.05rem; box-shadow: 0 10px 24px rgba(62, 45, 28, 0.04); }
    .kpi-label { color: #7a685d; font-size: 0.93rem; margin-bottom: 0.22rem; }
    .kpi-value { color: #302925; font-size: 1.72rem; font-weight: 900; line-height: 1.1; }
    .kpi-sub { color: #8a7a70; font-size: 0.9rem; margin-top: 0.25rem; }
    
    .guide-card { border-left: 5px solid #ccaa75; background: #fffdfa; border-radius: 16px; border: 1px solid #efe4d6; border-left-width: 5px; padding: 1rem; margin-bottom: 0.7rem; line-height: 1.65; }
    .info-card { border-radius: 18px; background: #fffaf4; border: 1px solid #eadfce; padding: 1rem; margin-bottom: 0.75rem; line-height: 1.65; }
    .risk-box { border-radius: 24px; padding: 1.25rem 1.3rem; border: 1px solid #ebe1d2; background: linear-gradient(180deg, #fffdf9 0%, #fbf6ee 100%); box-shadow: 0 12px 28px rgba(62, 45, 28, 0.05); }
    .eyebrow { display: inline-block; font-weight: 800; color: #9a5d55; background: rgba(255,255,255,0.8); border: 1px solid #eaded0; padding: 0.32rem 0.74rem; border-radius: 999px; margin-bottom: 0.85rem; font-size: 0.88rem; }
    .muted { color: #7c7068; font-size: 0.93rem; line-height: 1.65; }
</style>
"""

# ==========================================
# 4. 유틸리티 함수
# ==========================================
def weighted_rate(series: pd.Series, weights: pd.Series, positive: str) -> float:
    mask = series.notna() & weights.notna()
    if mask.sum() == 0: return np.nan
    s, w = series[mask], weights[mask]
    total = w.sum()
    if total == 0: return np.nan
    return float(w[s == positive].sum() / total)

def weighted_distribution(df: pd.DataFrame, col: str, weight_col: str = "가중치") -> pd.DataFrame:
    tmp = df[[col, weight_col]].dropna().copy()
    out = tmp.groupby(col, as_index=False)[weight_col].sum().rename(columns={weight_col: "가중치합"})
    out["비율"] = out["가중치합"] / out["가중치합"].sum()
    return out.sort_values("비율", ascending=False)

def risk_label(prob: float, threshold: float = 0.4996) -> str:
    if prob >= threshold: return "높음"
    if prob >= (threshold * 0.8): return "보통"
    return "낮음"

def get_risk_colors(prob: float, threshold: float = 0.4996) -> tuple[str, str, str]:
    if prob >= threshold: return "#b42318", "#fbeae9", "#b42318" 
    if prob >= (threshold * 0.8): return "#b26a00", "#fdf2e3", "#b26a00" 
    return "#067647", "#e5f5ed", "#067647" 

# ==========================================
# 5. 전처리된 데이터 로드 (고속화)
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_map_data(uploaded_file=None) -> pd.DataFrame:
    # 1. 전처리된 CSV 로드
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DATA_PATH)
    
    # 2. 대시보드 시각화용 한글 매핑 (직관적인 이름으로 덮어쓰기)
    df_vis = df.copy()
    
    map_dep = {"No": "일반군", "Risk": "위험군"}
    map_oral = {"No": "없음", "Yes": "있음"}
    map_sleep = {"No": "충분", "Yes": "부족"}
    map_time = {"≤3": "3시간 이하", "3 ~ 5": "3~5시간", "5 ~ 8": "5~8시간", "≥8": "8시간 이상"}
    
    df_vis["스마트폰 의존"] = df_vis["smartphone_dependence"].map(map_dep)
    df_vis["구강 증상"] = df_vis["oral_health_score"].apply(lambda x: "없음" if x == 0 else "있음")
    df_vis["수면 상태"] = df_vis["sleep_quality"].map(map_sleep)
    df_vis["주중 스마트폰 사용"] = df_vis["smartphone_use_day"].map(map_time)
    df_vis["주말 스마트폰 사용"] = df_vis["smartphone_use_weekend"].map(map_time)
    df_vis["가중치"] = df_vis["W"]
    
    return df_vis

# ==========================================
# 6. 모델 로드 및 예측 관련 함수
# ==========================================
# joblib.load() 로 scaler.pkl을 불러오기 위해 반드시 필요한 원본 클래스 구조
class WeightedStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X, sample_weight=None):
        pass
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X, sample_weight=None):
        return self.transform(X)

@st.cache_resource(show_spinner=False)
def load_ml_assets():
    model_path  = MODELS_DIR / "xgboost_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path   = MODELS_DIR / "model_meta.json"
    
    if not all(p.exists() for p in [model_path, scaler_path, meta_path]):
        return None, None, None
    
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta

def encode_for_xgboost(gender: str, school: str, grade: str, income: str, anxiety: str, stress: str, despair: str, suicidal: str, weekday: str, weekend: str, sp_group: str, sleep_group: str) -> np.ndarray:
    gen_enc  = {"여학생": 0, "남학생": 1}[gender]
    sch_enc  = {"중학교": 0, "고등학교": 1}[school]
    grd_enc  = {"하": 0, "중": 1, "상": 2}[grade]
    inc_enc  = {"하": 0, "중": 1, "상": 2}[income]
    anx_enc  = {"없음": 0, "약간": 1, "보통": 2, "심함": 3}[anxiety]
    str_enc  = {"낮음": 0, "중간": 1, "높음": 2}[stress]
    dsp_enc  = {"아니오": 0, "예": 1}[despair]
    sui_enc  = {"아니오": 0, "예": 1}[suicidal]
    day_enc  = {"3시간 이하": 0, "3~5시간": 1, "5~8시간": 2, "8시간 이상": 3}[weekday]
    we_enc   = {"3시간 이하": 0, "3~5시간": 1, "5~8시간": 2, "8시간 이상": 3}[weekend]
    dep_enc  = 1 if sp_group == "위험군" else 0
    slp_enc  = 1 if sleep_group == "부족" else 0
    
    return np.array([[gen_enc, sch_enc, grd_enc, inc_enc, anx_enc, str_enc, dsp_enc, sui_enc, day_enc, we_enc, dep_enc, slp_enc]])

# ==========================================
# 7. UI 구성 요소 (Tabs)
# ==========================================
def render_header(df: pd.DataFrame) -> None:
    oral_rate = weighted_rate(df["구강 증상"], df["가중치"], "있음") * 100
    dep_rate = weighted_rate(df["스마트폰 의존"], df["가중치"], "위험군") * 100
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">데이터 기반 청소년 구강 건강 관리 플랫폼</div>
                <h1 class="hero-title">{APP_TITLE}</h1>
                <p class="hero-subtitle">{APP_SUBTITLE}</p>
                <div class="hero-badges">
                    <span class="badge">스마트폰 과의존</span>
                    <span class="badge">구강 예측하기</span>
                    <span class="badge">맞춤형 솔루션</span>
                    <span class="badge">자료실</span>
                </div>
            </div>
            <div class="hero-panel">
                <div class="hero-panel-title">핵심 현황 (선행 데이터 요약)</div>
                <div class="mini-stat-grid">
                    <div class="mini-stat"><div class="mini-stat-label">스마트폰 의존 위험군</div><div class="mini-stat-value">{dep_rate:.1f}%</div></div>
                    <div class="mini-stat"><div class="mini-stat-label">구강 증상 경험</div><div class="mini-stat-value">{oral_rate:.1f}%</div></div>
                    <div class="mini-stat"><div class="mini-stat-label">분석 대상</div><div class="mini-stat-value">{len(df):,}명</div></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def prediction_tab(xgb_model, xgb_scaler, xgb_meta) -> None:
    st.markdown('<div class="section-title">구강 건강 위험도 예측 진단</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">아래의 설문을 작성하시면 스마트폰 의존 점수와 구강 건강 위험도를 즉시 계산해 드립니다.</div>', unsafe_allow_html=True)

    if xgb_model is None:
        st.error("⚠️ XGBoost 모델 폴더(`./models`)에서 파일들을 찾을 수 없습니다. 모델 학습 스크립트를 먼저 실행해 주세요.")
        return

    with st.form("comprehensive_diagnosis"):
        st.markdown("#### 👤 기본 정보")
        d1, d2, d3, d4 = st.columns(4)
        with d1: gender = st.selectbox("성별", ["여학생", "남학생"])
        with d2: school = st.selectbox("학교급", ["중학교", "고등학교"])
        with d3: grade  = st.selectbox("학업성적", ["상", "중", "하"])
        with d4: income = st.selectbox("경제수준", ["상", "중", "하"])

        anxiety, stress, despair, suicidal = "없음", "낮음", "아니오", "아니오"
        st.divider()

        st.markdown("#### 📱 스마트폰 과의존 척도 (S-Scale)")
        st.info("각 문항에 대해 자신에게 해당되는 정도를 체크해 주세요. (1점: 전혀 그렇지 않다 ~ 4점: 매우 그렇다)")
        
        sp_scores = []
        q_cols = st.columns(2)
        for i, q_text in enumerate(SMARTPHONE_QUESTIONS):
            with q_cols[i % 2]:
                sp_scores.append(st.radio(f"{i+1}. {q_text}", [1, 2, 3, 4], horizontal=True, key=f"q_{i}"))
        
        st.divider()

        st.markdown("#### 💤 수면의 질 질문")
        sleep_val = st.radio(
            "최근 7일 동안, 잠을 잔 시간이 피로회복에 충분하다고 생각합니까?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1:"매우 충분하다", 2:"충분하다", 3:"그저 그렇다", 4:"충분하지 않다", 5:"전혀 충분하지 않다"}[x],
            horizontal=True
        )

        st.divider()

        st.markdown("#### ⏱️ 스마트폰 사용 시간")
        c1, c2 = st.columns(2)
        with c1: weekday = st.selectbox("주중 스마트폰 사용 시간", TIME_OPTIONS)
        with c2: weekend = st.selectbox("주말 스마트폰 사용 시간", TIME_OPTIONS)

        predict_submitted = st.form_submit_button("진단 결과 확인하기", type="primary", width="stretch")

    if predict_submitted:
        total_sp_score = sum(sp_scores)
        sp_group = "위험군" if total_sp_score >= 23 else "일반군"
        sleep_group = "부족" if sleep_val >= 3 else "충분"

        X_raw = encode_for_xgboost(gender, school, grade, income, anxiety, stress, despair, suicidal, weekday, weekend, sp_group, sleep_group)
        X_scaled = xgb_scaler.transform(X_raw)
        prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
        
        # 모델의 Youden's 임계값 가져오기
        threshold = xgb_meta.get("youden_threshold", 0.4996)
        risk = risk_label(prob, threshold)
        
        st.session_state["diag_sp_group"] = sp_group
        st.session_state["diag_oral_risk"] = risk
        
        t_color, bg_color, b_color = get_risk_colors(prob, threshold)

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.metric("나의 스마트폰 의존 점수", f"{total_sp_score}점", delta=f"{sp_group}", delta_color="inverse")
            st.write(f"**진단:** 당신은 스마트폰 **{sp_group}**에 해당합니다.")
        
        with res_c2:
            st.markdown(f"""
            <div class="risk-box" style="text-align:center; background:{bg_color}; border:2px solid {b_color}; box-shadow: 0 8px 20px {t_color}25;">
                <div style="color:{t_color}; opacity:0.85; font-size:0.95rem; font-weight:700;">구강 건강 위험도 예측 (AI 분석)</div>
                <h2 style="color:{t_color}; font-weight:900; margin: 0.5rem 0;">{risk}</h2>
                <p style="color:{t_color}; font-weight:700; margin:0;">발생 확률: {prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.success("진단이 완료되었습니다! 옆의 '💡 맞춤형 솔루션' 탭으로 이동하여 나에게 딱 맞는 가이드를 확인해 보세요!")

def medical_tab(key_suffix: str = "main") -> None:
    st.markdown('<div class="section-title">의료기관 연계</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">구강 증상이 지속되거나 관리가 필요한 경우 주변 치과를 확인하고 예약 요청까지 연결할 수 있도록 구성했습니다.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.06, 0.94], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### 주변 치과 찾기")
            location = st.text_input("지역 또는 주소", placeholder="예: 서울 성동구 / 왕십리", key=f"loc_{key_suffix}")
            keyword = st.selectbox("검색 유형", ["치과", "청소년 치과", "예방 진료 치과", "야간 진료 치과"], key=f"kw_{key_suffix}")
            if st.button("주변 치과 검색", width="stretch", key=f"btn_search_{key_suffix}"):
                query = quote_plus(f"{location} {keyword}".strip())
                st.markdown(f"[지도에서 확인하기](https://www.google.com/maps/search/?api=1&query={query})")
        with st.container(border=True):
            st.markdown("#### 예약 요청")
            name = st.text_input("이름", key=f"r_name_{key_suffix}")
            st.date_input("희망 날짜", key=f"r_date_{key_suffix}")
            st.text_area("증상 또는 요청 사항", placeholder="예: 치아 통증, 잇몸 출혈, 스케일링 문의", key=f"r_note_{key_suffix}")
            if st.button("예약 요청 저장", type="primary", width="stretch", key=f"btn_save_{key_suffix}"):
                if name: st.success(f"{name}님의 예약 요청이 저장되었습니다.")
                else: st.warning("이름을 입력해 주세요.")
    with c2:
        st.markdown("""
        <div class="card">
            <h4 style="margin-top:0;">진료 연결 안내</h4>
            <div class="guide-card">예측 결과가 높음 수준이거나 통증, 출혈, 씹기 불편이 반복되면 진료 연계를 우선 고려합니다.</div>
            <div class="guide-card">진료 전에는 증상 발생 시점, 빈도, 불편한 부위를 정리해 두면 상담에 도움이 됩니다.</div>
            <div class="guide-card">예방 목적의 방문인 경우에도 스케일링, 검진 주기, 생활습관 상담을 함께 확인할 수 있습니다.</div>
        </div>
        """, unsafe_allow_html=True)

def solution_tab() -> None:
    st.markdown('<div class="section-title">개인 맞춤형 솔루션</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">진단 결과에 맞는 1:1 맞춤형 행동 가이드를 제안해 드립니다.</div>', unsafe_allow_html=True)

    user_sp = st.session_state.get("diag_sp_group", None)
    user_oral = st.session_state.get("diag_oral_risk", None)

    user_type = 0
    if user_sp and user_oral:
        is_sp_risk = (user_sp == "위험군")
        is_oral_risk = (user_oral in ["높음", "보통"])
        
        if is_sp_risk and is_oral_risk: user_type = 1
        elif not is_sp_risk and is_oral_risk: user_type = 2
        elif is_sp_risk and not is_oral_risk: user_type = 3
        elif not is_sp_risk and not is_oral_risk: user_type = 4

    if user_type == 0:
        st.info("💡 먼저 **'🎯 구강 예측하기'** 탭에서 진단을 진행하시면, 나에게 딱 맞는 맞춤 솔루션을 확인할 수 있습니다.")
        return

    cards = {
        1: """
            <div class="card" style="border: 3px solid #b42318; box-shadow: 0 10px 25px #b4231840;">
                <h4 style="color:#b42318; margin-top:0; font-size: 1.4rem;">🔴 집중 케어 솔루션</h4>
                <div class="eyebrow" style="background:#fbeae9; color:#b42318; border:none;">☑ 스마트폰 의존 (위험군)   ☑ 구강 위험도</div>
                <p class="muted" style="font-size: 1.05rem;">스마트폰 과사용이 수면 부족과 구강 관리 소홀로 이어져 치아 건강에 적신호가 켜진 상태입니다. 두 가지 모두 적극적인 관리가 필요합니다.</p>
                <div style="display: flex; gap: 1rem; margin-top: 1.2rem;">
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>1. 디지털 디톡스:</b><br>취침 1시간 전 스마트폰 전원을 끄고 다른 방에 두세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>2. 구강 리듬 회복:</b><br>스마트폰을 보며 간식을 먹는 습관을 중단하고, 식후 3분 이내 양치하세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>3. 즉각적인 치과 방문:</b><br>불편감이 생겼다면 미루지 말고 진료를 받아야 합니다.</div>
                </div>
            </div>""",
        2: """
            <div class="card" style="border: 3px solid #b26a00; box-shadow: 0 10px 25px #b26a0040;">
                <h4 style="color:#b26a00; margin-top:0; font-size: 1.4rem;">🟡 구강 집중 솔루션</h4>
                <div class="eyebrow" style="background:#fdf2e3; color:#b26a00; border:none;">☐ 스마트폰 의존 (일반군)   ☑ 구강 위험도</div>
                <p class="muted" style="font-size: 1.05rem;">스마트폰 사용은 훌륭하게 통제하고 있으나, 식습관이나 양치 습관 등 다른 요인으로 인해 구강 건강이 위협받고 있습니다.</p>
                <div style="display: flex; gap: 1rem; margin-top: 1.2rem;">
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>1. 사각지대 점검:</b><br>칫솔질만으로는 부족할 수 있습니다. 치실 및 치간칫솔 사용을 습관화하세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>2. 식습관 개선:</b><br>당류, 탄산음료 섭취를 줄이고 섭취 후에는 반드시 물로 입을 헹구세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>3. 정기 검진:</b><br>증상이 악화되기 전에 스케일링 및 구강 검진을 받으세요.</div>
                </div>
            </div>""",
        3: """
            <div class="card" style="border: 3px solid #005a9e; box-shadow: 0 10px 25px #005a9e40;">
                <h4 style="color:#005a9e; margin-top:0; font-size: 1.4rem;">🔵 스마트폰 디톡스 솔루션</h4>
                <div class="eyebrow" style="background:#e6f0fa; color:#005a9e; border:none;">☑ 스마트폰 의존 (위험군)   ☐ 구강 위험도</div>
                <p class="muted" style="font-size: 1.05rem;">현재 구강 건강은 양호하지만, 스마트폰 의존이 지속되면 수면 및 생활 리듬이 깨져 구강 건강까지 나빠질 수 있습니다.</p>
                <div style="display: flex; gap: 1rem; margin-top: 1.2rem;">
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>1. 사용 시간 제한:</b><br>앱 타이머 기능을 활용해 주중/주말 스마트폰 사용 시간을 서서히 줄여보세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>2. 수면 질 향상:</b><br>침대에 누울 때는 스마트폰을 멀리 두어 수면의 질을 높이세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>3. 현재 습관 유지:</b><br>지금의 꼼꼼한 구강 관리 습관은 계속해서 잃지 않도록 주의하세요.</div>
                </div>
            </div>""",
        4: """
            <div class="card" style="border: 3px solid #067647; box-shadow: 0 10px 25px #06764740;">
                <h4 style="color:#067647; margin-top:0; font-size: 1.4rem;">🟢 건강 유지 솔루션</h4>
                <div class="eyebrow" style="background:#e5f5ed; color:#067647; border:none;">☐ 스마트폰 의존 (일반군)   ☐ 구강 위험도</div>
                <p class="muted" style="font-size: 1.05rem;">스마트폰 사용도 스스로 잘 조절하고 있으며, 구강 건강도 매우 안전하고 훌륭하게 유지되고 있는 최상의 상태입니다!</p>
                <div style="display: flex; gap: 1rem; margin-top: 1.2rem;">
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>1. 루틴 지키기:</b><br>현재의 규칙적인 생활과 충분한 수면 리듬을 흔들림 없이 유지하세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>2. 예방적 관리:</b><br>특별한 증상이 없더라도 6개월~1년 주기로 예방 목적의 치과 방문을 하세요.</div>
                    <div class="guide-card" style="flex: 1; margin-bottom:0;"><b>3. 긍정적 강화:</b><br>본인의 올바른 생활 습관 노하우를 주변 친구나 가족과 함께 공유해 보세요.</div>
                </div>
            </div>"""
    }
    
    st.markdown(cards[user_type], unsafe_allow_html=True)

    if user_type in [1, 2]:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.divider()
        medical_tab(key_suffix="solution_view")

def render_kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def dashboard_tab(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">통계 예측지도 (EDA 심층 분석)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">업로드된 분석 결과를 반영하여, 스마트폰 이용 시간과 수면의 질이 구강 건강에 미치는 영향을 직관적으로 시각화했습니다.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi("스마트폰 의존 위험군", f"{weighted_rate(df['스마트폰 의존'], df['가중치'], '위험군')*100:.1f}%", "가중치 반영")
    with c2: render_kpi("수면 부족군", f"{weighted_rate(df['수면 상태'], df['가중치'], '부족')*100:.1f}%", "보조 지표")
    with c3: render_kpi("구강 증상 경험", f"{weighted_rate(df['구강 증상'], df['가중치'], '있음')*100:.1f}%", "핵심 결과지표")
    with c4: render_kpi("선정 모델", "XGBoost v4", "최적화 예측")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="hero-panel-title">📊 스마트폰 사용시간 대비 구강 증상 발생 비율 (가중치 반영)</div>', unsafe_allow_html=True)
    
    def plot_weighted_crosstab_bar(time_col: str, title: str):
        grouped = df.groupby([time_col, '구강 증상'])['가중치'].sum().reset_index()
        totals = grouped.groupby(time_col)['가중치'].transform('sum')
        grouped['비율(%)'] = (grouped['가중치'] / totals) * 100
        grouped[time_col] = pd.Categorical(grouped[time_col], categories=TIME_OPTIONS, ordered=True)
        grouped = grouped.sort_values(time_col)
        fig = px.bar(grouped, x=time_col, y='비율(%)', color='구강 증상', title=title, color_discrete_map={"없음": "#133F52", "있음": "#27770F"}, text=grouped['비율(%)'].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition='inside', textfont_size=13, textfont_color="white")
        fig.update_layout(barmode='stack', height=400, margin=dict(t=40, b=20, l=10, r=10), yaxis_title="비율 (%)", xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    r0_c1, r0_c2 = st.columns(2, gap="large")
    with r0_c1: st.plotly_chart(plot_weighted_crosstab_bar('주중 스마트폰 사용', '주중 스마트폰 사용시간 vs 구강문제'), use_container_width=True)
    with r0_c2: st.plotly_chart(plot_weighted_crosstab_bar('주말 스마트폰 사용', '주말 스마트폰 사용시간 vs 구강문제'), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1_c1, r1_c2 = st.columns(2, gap="large")
    with r1_c1:
        st.markdown('<div class="hero-panel-title">📱 스마트폰 의존도 비율</div>', unsafe_allow_html=True)
        dist_dep = weighted_distribution(df, "스마트폰 의존")
        dist_dep["표시비율"] = dist_dep["비율"] * 100
        fig_pie = px.pie(dist_dep, names="스마트폰 의존", values="표시비율", hole=0.45, color="스마트폰 의존", color_discrete_map={"일반군": "#64B5F6", "위험군": "#E57373"})
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14, textfont_weight="bold", marker=dict(line=dict(color='#fffdf9', width=3)))
        fig_pie.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1_c2:
        st.markdown('<div class="hero-panel-title">📈 이용 시간별 구강 증상 경험률 추이</div>', unsafe_allow_html=True)
        def get_time_oral_risk(col_name, label):
            res = []
            for t in TIME_OPTIONS:
                sub = df[df[col_name] == t]
                if len(sub) > 0:
                    res.append({"사용시간": t, "구강증상 경험률(%)": weighted_rate(sub["구강 증상"], sub["가중치"], "있음") * 100, "구분": label})
            return res
        df_time_oral = pd.DataFrame(get_time_oral_risk("주중 스마트폰 사용", "주중") + get_time_oral_risk("주말 스마트폰 사용", "주말"))
        fig_line = px.line(df_time_oral, x="사용시간", y="구강증상 경험률(%)", color="구분", markers=True, color_discrete_sequence=["#5C6BC0", "#FFA726"])
        fig_line.update_traces(line=dict(width=3.5), marker=dict(size=9, line=dict(width=2, color="white")))
        fig_line.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="경험률 (%)", xaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e8dfd2')
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="hero-panel-title">💤 수면 상태 & 스마트폰 의존 복합 위험도</div>', unsafe_allow_html=True)
    inter_res = []
    for dep in ["일반군", "위험군"]:
        for slp in ["충분", "부족"]:
            sub = df[(df["스마트폰 의존"] == dep) & (df["수면 상태"] == slp)]
            if len(sub) > 0:
                inter_res.append({"의존도": dep, "수면": slp, "구강증상 경험률(%)": weighted_rate(sub["구강 증상"], sub["가중치"], "있음") * 100})
    df_inter = pd.DataFrame(inter_res)
    fig_bar = px.bar(df_inter, x="의존도", y="구강증상 경험률(%)", color="수면", barmode="group", text="구강증상 경험률(%)", color_discrete_map={"충분": "#81C784", "부족": "#F06292"})
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=13, textfont_weight="bold")
    fig_bar.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="경험률 (%)", xaxis_title="", yaxis_range=[0, max(df_inter["구강증상 경험률(%)"])*1.2], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_bar, use_container_width=True)

def guide_tab() -> None:
    st.markdown('<div class="section-title">관리 가이드</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">예측 결과를 실제 행동으로 연결할 수 있도록 솔루션 제안 구조로 구성했습니다.</div>', unsafe_allow_html=True)
    
    subtabs = st.tabs(["생활습관 관리", "구강 관리"])
    with subtabs[0]:
        left, right = st.columns([1.08, 0.92], gap="large")
        with left: 
            st.markdown("""
            <div class="card">
                <h4 style="margin-top:0;">생활습관 관리 가이드</h4>
                <div class="guide-card"><b>스마트폰 사용 관리</b><br>취침 전 30분은 화면 노출을 줄이고, 주중·주말 사용 시간을 일정하게 관리합니다.</div>
                <div class="guide-card"><b>수면 리듬 관리</b><br>기상 시간과 취침 시간을 일정하게 유지하고, 수면 부족 상태가 이어지지 않도록 생활 리듬을 점검합니다.</div>
                <div class="guide-card"><b>식습관 및 수분 관리</b><br>당류·탄산 섭취 이후에는 물을 마시고, 야식 및 불규칙한 간식 습관을 줄입니다.</div>
            </div>""", unsafe_allow_html=True)
        with right: 
            st.markdown("""
            <div class="card">
                <h4 style="margin-top:0;">근거</h4>
                <div class="info-card">스마트폰 의존 여부에 따른 수면 충분 여부와 구강 증상 여부를 함께 비교하면 생활습관 상태를 다각도로 확인할 수 있습니다.</div>
                <div class="info-card">앱의 분석 구조는 스마트폰 의존 여부를 기준으로 수면 상태와 구강 증상 여부를 각각 해석하도록 설계되어 있습니다.</div>
                <div class="info-card">따라서 생활습관 관리 탭에서는 스마트폰 사용, 수면, 구강 관리 실천을 함께 제안하여 행동 변화로 이어지도록 구성합니다.</div>
            </div>""", unsafe_allow_html=True)

    with subtabs[1]:
        c1, c2 = st.columns(2, gap="large")
        with c1: 
            st.markdown("""
            <div class="card">
                <h4 style="margin-top:0;">주요 구강 정보</h4>
                <div class="guide-card"><b>치아 통증</b><br>치아가 욱신거리거나 음식을 씹을 때 통증이 느껴지는 증상입니다.</div>
                <div class="guide-card"><b>잇몸 출혈</b><br>양치 중 또는 평소 잇몸에서 피가 나는 증상으로 관리가 필요합니다.</div>
                <div class="guide-card"><b>치아 파절·불편감</b><br>치아가 깨지거나 씹을 때 불편감이 반복되면 점검이 필요합니다.</div>
            </div>""", unsafe_allow_html=True)
        with c2: 
            st.markdown("""
            <div class="card">
                <h4 style="margin-top:0;">구강 관리 방법</h4>
                <div class="guide-card"><b>기본 위생 관리</b><br>하루 2회 이상 양치, 치실 사용, 식후 물 마시기를 기본 루틴으로 유지합니다.</div>
                <div class="guide-card"><b>증상 관찰</b><br>시림, 통증, 출혈, 입냄새가 반복될 경우 증상을 기록하고 진료 필요 여부를 확인합니다.</div>
                <div class="guide-card"><b>예방 관리</b><br>증상이 없더라도 정기 검진과 스케일링 주기를 관리합니다.</div>
            </div>""", unsafe_allow_html=True)

def program_tab() -> None:
    st.markdown('<div class="section-title">자료실</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">가정, 학교, 지역사회에서 활용할 수 있는 다양한 예방 교육 및 치유 프로그램을 안내해 드립니다.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <h4 style="color:#2d2724; margin-top:0;">🏕️ 가족치유캠프</h4>
            <p class="muted">스마트폰 사용 문제로 갈등을 겪는 가족이 함께 참여하여 소통을 강화하고 관계를 회복하는 단기 체험형 캠프입니다.</p>
        </div>
        <div class="card">
            <h4 style="color:#2d2724; margin-top:0;">👨‍👩‍👧 보호자 교육</h4>
            <p class="muted">자녀의 올바른 스마트폰 사용 지도법과 더불어, 디지털 환경에서의 구강 건강 관리법을 안내하는 학부모 대상 온/오프라인 교육 자료입니다.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <h4 style="color:#2d2724; margin-top:0;">🏠 기숙치유캠프</h4>
            <p class="muted">스마트폰 과의존 위험군 청소년을 대상으로 전문가의 집중 상담과 대안 활동을 제공하는 장기 기숙형 치유 프로그램입니다.</p>
        </div>
        <div class="card">
            <h4 style="color:#2d2724; margin-top:0;">🏫 예방교육</h4>
            <p class="muted">학교 및 유관 기관에서 자체적으로 활용할 수 있는 연령별 맞춤형 스마트폰 과의존 예방 및 디지털 리터러시 교육 콘텐츠 및 교안입니다.</p>
        </div>""", unsafe_allow_html=True)

# ==========================================
# 8. 메인 로직
# ==========================================
def main() -> None:
    # 데이터 로드
    try:
        df = load_and_map_data()
    except FileNotFoundError:
        st.warning("⚠️ 전처리된 데이터(`kyrbs2020_clean_v1.csv`)를 찾을 수 없습니다. 직접 업로드 해주세요.")
        uploaded_file = st.file_uploader("CSV 데이터 업로드", type=["csv"])
        if uploaded_file is None: st.stop()
        df = load_and_map_data(uploaded_file)
    
    # 모델 로드
    xgb_model, xgb_scaler, xgb_meta = load_ml_assets()

    # 상단 헤더 렌더링
    render_header(df)
    
    tabs = st.tabs(["📱 스마트폰 과의존", "🎯 구강 예측하기", "💡 맞춤형 솔루션", "🏫 자료실"])

    with tabs[0]: 
        st.markdown('<div class="section-title">스마트폰 과의존 이해하기</div><div class="section-desc">스마트폰 과의존의 정의와 진단 기준을 확인해 보세요.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <h4 style="color:#b55a5b; margin-top:0;">💡 스마트폰 과의존이란?</h4>
            <p class="muted">스마트폰을 과도하게 이용함에 따라 조절력이 감소하고, 일상생활에서 문제적 결과를 경험하게 되는 상태를 의미합니다.</p>
            <hr style="border:0; border-top:1px solid #e8dfd2; margin: 1.2rem 0;">
            <h4 style="font-size:1.1rem; margin-top:0;">🔍 3대 핵심 현상</h4>
            <div style="display: flex; gap: 1rem;">
                <div class="guide-card" style="flex: 1; margin-bottom: 0;"><b>1. 조절실패 (Control Failure)</b><br><span style="font-size:0.9rem; color:#666;">이용시간 조절이 어렵고 계획대로 되지 않음</span></div>
                <div class="guide-card" style="flex: 1; margin-bottom: 0;"><b>2. 현저성 (Salience)</b><br><span style="font-size:0.9rem; color:#666;">일상에서 스마트폰 이용이 가장 중요한 활동이 됨</span></div>
                <div class="guide-card" style="flex: 1; margin-bottom: 0;"><b>3. 문제적 결과 (Problematic Consequences)</b><br><span style="font-size:0.9rem; color:#666;">신체적·심리적·사회적으로 부정적 결과 경험함</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <h4 style="color:#b55a5b; margin-top:0;">📊 청소년 진단 기준 (S-Scale)</h4>
            <p class="muted">본 서비스는 한국지능정보사회진흥원(NIA)의 표준 척도를 기준으로 분류합니다.</p>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <div class="info-card" style="flex: 1; border-left: 5px solid #ff4b4b; margin-bottom: 0;"><b style="color:#ff4b4b;">고위험군 (31점 이상)</b><br><br><span style="font-size:0.9rem; color:#666;">통제력을 상실하여 일상생활에 심각한 장애가 있으며 대인관계나 건강에 문제가 발생한 상태</span></div>
                <div class="info-card" style="flex: 1; border-left: 5px solid #ffa500; margin-bottom: 0;"><b style="color:#ffa500;">잠재적 위험군 (23~30점)</b><br><br><span style="font-size:0.9rem; color:#666;">조절력이 약화되어 일상생활에 지장이 시작되거나 심리적 불안을 느끼는 상태</span></div>
                <div class="info-card" style="flex: 1; border-left: 5px solid #28a745; margin-bottom: 0;"><b style="color:#28a745;">일반군 (22점 이하)</b><br><br><span style="font-size:0.9rem; color:#666;">스마트폰 이용을 스스로 적절히 조절할 수 있으며 문제적 결과가 거의 없는 상태</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <h4 style="color:#b55a5b; margin-top:0;">🛡️ 상태별 분류 및 조치 가이드</h4>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <div class="risk-box" style="flex: 1; border-left: 5px solid #ff4b4b; border-top: none;"><h5 style="margin-top:0; color:#ff4b4b;">고위험군 조치</h5><p class="muted" style="margin-bottom:0; font-size:0.9rem;">전문적인 상담과 치료가 권장됩니다. 스마트폰 사용을 전면적으로 통제하고 디지털 디톡스 환경을 조성해야 하며, 학교 및 전문 기관의 적극적인 개입이 필요할 수 있습니다.</p></div>
                <div class="risk-box" style="flex: 1; border-left: 5px solid #ffa500; border-top: none;"><h5 style="margin-top:0; color:#ffa500;">잠재적 위험군 조치</h5><p class="muted" style="margin-bottom:0; font-size:0.9rem;">스스로 사용 시간을 점검하고 제한하는 규칙을 세워야 합니다. 가족이나 친구의 도움을 받아 사용 패턴을 모니터링하고 대체 활동(운동, 취미 등)을 늘리는 것이 좋습니다.</p></div>
                <div class="risk-box" style="flex: 1; border-left: 5px solid #28a745; border-top: none;"><h5 style="margin-top:0; color:#28a745;">일반군 조치</h5><p class="muted" style="margin-bottom:0; font-size:0.9rem;">현재의 건강한 스마트폰 사용 습관을 꾸준히 유지하세요. 정기적으로 스스로의 사용량을 점검하고, 수면 시간 전에는 기기 사용을 자제하는 기본 수칙을 지켜주세요.</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **'🎯 구강 예측하기'** 탭으로 이동하여 나의 상태를 직접 진단해 보세요!")
        
    with tabs[1]:
        prediction_tab(xgb_model, xgb_scaler, xgb_meta)
        
    with tabs[2]:
        solution_tab()
        
    with tabs[3]:
        resource_subtabs = st.tabs(["📊 통계 예측지도", "📘 관리 가이드", "프로그램 안내"])
        with resource_subtabs[0]: dashboard_tab(df)
        with resource_subtabs[1]: guide_tab()
        with resource_subtabs[2]: program_tab()
            
if __name__ == "__main__" :
    main()