"""
[목적] 전처리된 KYRBS 2020 데이터를 기반으로 복합표본 통계 분석을 수행합니다.
[분석 1] Rao-Scott 카이제곱 검정 — 구강 건강과 각 독립변수 간 연관성 검정
[분석 2] 위계적 다중 로지스틱 회귀분석 — 3단계 블록 투입
[입력] data/processed/kyrbs2020_clean_v1.csv
[출력] data/processed/다중인덱스_가중치교차분석표_oral_health_2020.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, distributions
import warnings
warnings.filterwarnings('ignore')

df_clean = pd.read_csv('./data/processed/kyrbs2020_clean_v1.csv')
WEIGHT_COL = 'W'

# =====================================================================
# [매핑] 영문 범주값 → 한국어 표시 이름
# =====================================================================
VALUE_MAPPING = {
    'Male': '남자', 'Female': '여자',
    'Middle school': '중학교', 'High school': '고등학교',
    'High': '상', 'Middle': '중', 'Low': '하',
    '≤3': '3시간 이하', '3 ~ 5': '3~5시간', '5 ~ 8': '5~8시간', '≥8': '8시간 이상',
    'No': '없음(아니오)', 'Yes': '있음(예)', 'Risk': '위험군',
    'Mild': '가벼움', 'Moderate': '중간', 'Severe': '심각'
}

# [순서] 범주값 정렬 기준 리스트
CATEGORY_ORDER = [
    'Male', 'Female', 'Middle school', 'High school',
    'High', 'Middle', 'Low',
    '≤3', '3 ~ 5', '5 ~ 8', '≥8',
    'No', 'Mild', 'Moderate', 'Severe', 'Risk', 'Yes'
]


# =====================================================================
# [함수] Rao-Scott 1차 보정 카이제곱 검정
# =====================================================================
def rao_scott_chi2(ct_w: pd.DataFrame, weights: np.ndarray) -> tuple[float, float]:
    """
    [역할] 복합표본설계 특성을 반영한 Rao-Scott 1차 보정 카이제곱 검정을 수행합니다.
    [원리] 실효 표본 수(n_eff)로 정규화 후 설계효과(DEFF)로 X²·자유도 동시 보정
    [입력] ct_w     — 가중치 합산 교차표 (pd.crosstab 결과)
           weights  — 개별 관측치 가중치 배열
    [출력] (x2_rs, p_rs) — 보정된 카이제곱 통계량과 p-값
    [참고] 일반 카이제곱 대비 1종 오류를 줄여 복합표본에 적합
    """
    W     = weights.sum()
    n_eff = (W ** 2) / (weights ** 2).sum()   # [실효 표본 수] 설계효과 보정의 분모
    deff  = len(weights) / n_eff               # [설계효과 DEFF] 실제 n / 실효 n

    x2, _, dof, _ = chi2_contingency(ct_w * (n_eff / W))
    x2_rs = x2 / deff
    df_rs = max(dof / deff, 0.5)
    p_rs  = float(distributions.chi2.sf(x2_rs, df_rs))

    return x2_rs, p_rs


# =====================================================================
# [함수] 가중치 반영 교차분석표 생성
# =====================================================================
def generate_crosstab(
    df, var_col, var_name_ko, target_col, weight_col,
    target_val_0, target_val_1, target_label_0, target_label_1
) -> pd.DataFrame:
    """
    [역할] 독립변수와 타깃 변수의 가중치 반영 교차분석표를 생성합니다.
    [입력] var_col      — 독립변수 컬럼명
           var_name_ko  — 독립변수 한국어 이름 (표 첫 행에 표시)
           target_col   — 종속변수 컬럼명
           target_val_0/1    — 종속변수의 두 범주값
           target_label_0/1  — 종속변수 표 헤더 레이블
    [출력] 가중치 반영 교차분석 결과 DataFrame
    """
    valid_mask = df[[var_col, target_col, weight_col]].notna().all(axis=1)
    df_valid   = df[valid_mask]
    weights    = df_valid[weight_col].values
    W_total    = weights.sum()

    ct_w = pd.crosstab(
        df_valid[var_col], df_valid[target_col],
        values=df_valid[weight_col], aggfunc='sum'
    ).fillna(0)

    x2_rs, p_rs = rao_scott_chi2(ct_w, weights)
    p_str       = '< 0.001' if p_rs < 0.001 else f'{p_rs:.3f}'

    # [정렬] CATEGORY_ORDER 기준으로 범주 정렬
    categories = sorted(ct_w.index.tolist(),
                        key=lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else 999)

    rows = []
    sum_w_0 = sum_w_1 = sum_w_total = 0

    for i, cat in enumerate(categories):
        w_0         = ct_w.loc[cat, target_val_0] if target_val_0 in ct_w.columns else 0
        w_1         = ct_w.loc[cat, target_val_1] if target_val_1 in ct_w.columns else 0
        w_cat_total = w_0 + w_1
        sum_w_0     += w_0;  sum_w_1 += w_1;  sum_w_total += w_cat_total

        cat_ratio = (w_cat_total / W_total) * 100
        pct_0     = (w_0 / w_cat_total) * 100 if w_cat_total > 0 else 0
        pct_1     = (w_1 / w_cat_total) * 100 if w_cat_total > 0 else 0

        rows.append({
            '변수'        : var_name_ko if i == 0 else '',
            '구분'        : VALUE_MAPPING.get(cat, cat),
            '표본 수'     : f'{w_cat_total:,.0f} ({cat_ratio:.1f}%)',
            target_label_0: f'{w_0:,.0f} ({pct_0:.1f}%)',
            target_label_1: f'{w_1:,.0f} ({pct_1:.1f}%)',
            '검정통계량'  : f'{x2_rs:.3f}' if i == 0 else '',
            'p-value'     : p_str         if i == 0 else '',
        })

    # [합계 행] 전체 합산 추가
    total_r0 = (sum_w_0 / sum_w_total) * 100 if sum_w_total > 0 else 0
    total_r1 = (sum_w_1 / sum_w_total) * 100 if sum_w_total > 0 else 0
    rows.append({'변수': 'total', '구분': '', '표본 수': f'{sum_w_total:,.0f}',
                 target_label_0: f'{sum_w_0:,.0f} ({total_r0:.1f}%)',
                 target_label_1: f'{sum_w_1:,.0f} ({total_r1:.1f}%)',
                 '검정통계량': '', 'p-value': ''})

    return pd.DataFrame(rows)


# =====================================================================
# [분석 1] Rao-Scott 교차분석표 생성 및 저장
# =====================================================================

# [변수 매핑] 분석 대상 독립변수와 한국어 이름
VAR_MAPPING = {
    'gender'               : '성별',
    'school'               : '학교',
    'grade'                : '성적수준',
    'income'               : '경제수준',
    'smartphone_use_day'   : '주중 스마트폰 사용시간',
    'smartphone_use_weekend': '주말 스마트폰 사용시간',
    'smartphone_dependence' : '스마트폰 의존도',
    'sleep_quality'         : '수면 품질',
    'anxiety'               : '불안',
    'stress'                : '스트레스',
    'despair'               : '절망감',
    'suicidal_thoughts'     : '자살 생각',
}

# [타깃] 구강 건강 — oral_health_score를 이진 분류로 변환 (0=없음, >=1=있음)
df_clean['oral_health'] = df_clean['oral_health_score'].apply(lambda x: 'No' if x == 0 else 'Yes')
TARGET_COL              = 'oral_health'
VAL_0, VAL_1            = 'No', 'Yes'
LABEL_0, LABEL_1        = '좋음(0)', '나쁨(1)'

final_rows = []
for col_name, kor_name in VAR_MAPPING.items():
    temp_df = generate_crosstab(
        df_clean, col_name, kor_name,
        target_col=TARGET_COL, weight_col=WEIGHT_COL,
        target_val_0=VAL_0,    target_val_1=VAL_1,
        target_label_0=LABEL_0, target_label_1=LABEL_1,
    )
    final_rows.append(temp_df)

df_crosstab = pd.concat(final_rows, ignore_index=True)

# [멀티인덱스] 교차분석표 컬럼 구조 설정
multi_cols = [
    ('', '변수'), ('', '구분'), ('', '표본 수'),
    (TARGET_COL, LABEL_0), (TARGET_COL, LABEL_1),
    ('', '검정통계량(Rao-Scott)'), ('', 'p-value'),
]
df_crosstab.columns = pd.MultiIndex.from_tuples(multi_cols)
display(df_crosstab)

df_crosstab.to_csv(
    f'./data/processed/다중인덱스_가중치교차분석표_{TARGET_COL}_2020.csv',
    encoding='utf-8-sig'
)


# =====================================================================
# [분석 2] 위계적 다중 로지스틱 회귀분석
# =====================================================================
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

plt.rcParams["font.family"]        = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "./outputs_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [변수 그룹] 위계적 투입 순서
DEMO_VARS   = ["gender", "school", "grade", "income"]
MENTAL_VARS = ["anxiety", "stress", "despair", "suicidal_thoughts"]
SP_VAR      = ["smartphone_use_day", "smartphone_use_weekend", "smartphone_dependence"]
SL_VAR      = ["sleep_quality"]
REGRESSION_FEATS = DEMO_VARS + MENTAL_VARS + SP_VAR + SL_VAR
TARGET = "oral_poor"

# [인코딩] 범주형 → 순서형 정수 인코딩
VAR_KOR = {
    "gender": "성별", "school": "학교급", "grade": "학업성적",
    "income": "가구소득", "anxiety": "불안수준", "stress": "스트레스",
    "despair": "절망감", "suicidal_thoughts": "자살생각",
    "smartphone_use_day"    : "주중 스마트폰 사용시간",
    "smartphone_use_weekend": "주말 스마트폰 사용시간",
    "smartphone_dependence" : "스마트폰 의존도(위험=1)",
    "sleep_quality"         : "수면의 질(나쁨=1)",
}

ENCODE_MAP = {
    "gender"               : {"Female": 0, "Male": 1},
    "school"               : {"Middle school": 0, "High school": 1},
    "grade"                : {"Low": 0, "Middle": 1, "High": 2},
    "income"               : {"Low": 0, "Middle": 1, "High": 2},
    "anxiety"              : {"No": 0, "Mild": 1, "Moderate": 2, "Severe": 3},
    "stress"               : {"Low": 0, "Middle": 1, "High": 2},
    "despair"              : {"No": 0, "Yes": 1},
    "suicidal_thoughts"    : {"No": 0, "Yes": 1},
    "smartphone_use_day"   : {"≤3": 0, "3 ~ 5": 1, "5 ~ 8": 2, "≥8": 3},
    "smartphone_use_weekend": {"≤3": 0, "3 ~ 5": 1, "5 ~ 8": 2, "≥8": 3},
    "smartphone_dependence": {"No": 0, "Risk": 1},
    "sleep_quality"        : {"No": 0, "Yes": 1},
}

df_sm = df_clean.copy()
for col, mp in ENCODE_MAP.items():
    if col in df_sm.columns:
        df_sm[col] = df_sm[col].map(mp)

# [타깃] oral_health_score > 0이면 구강 문제 있음(1)
df_sm["oral_poor"] = (df_sm["oral_health_score"] > 0).astype(int)
df_sm = df_sm[REGRESSION_FEATS + [TARGET, WEIGHT_COL]].dropna().reset_index(drop=True)


# =====================================================================
# [VIF 검증] 다중공선성 확인
# =====================================================================
print("\n [1] 다중공선성(VIF) 검증")
X_vif    = pd.DataFrame(sm.add_constant(df_sm[REGRESSION_FEATS].astype(float)))
vif_data = pd.DataFrame({
    "변수명": [VAR_KOR.get(c, c) if c != "const" else "(상수항)" for c in X_vif.columns],
    "VIF"   : [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_data = vif_data[vif_data["변수명"] != "(상수항)"].sort_values("VIF", ascending=False).reset_index(drop=True)
display(vif_data.round(3))
print("  ※ VIF ≥ 10: 다중공선성 위험 / ≥ 5: 주의")


# =====================================================================
# [위계적 회귀] 3단계 블록 순차 투입
# =====================================================================
def format_pvalue(p: float) -> str:
    """[역할] p-값을 유의수준 표시 형식으로 변환합니다 (***/**/* 기준)."""
    if   p < 0.001: return "< .001 ***"
    elif p < 0.01:  return f"{p:.3f} **"
    elif p < 0.05:  return f"{p:.3f} *"
    else:           return f"{p:.3f}"

w_int = np.round(df_sm[WEIGHT_COL]).astype(int)
y_sm  = df_sm[TARGET]

# [블록 정의] 투입 순서: 인구학적+정신건강 → +스마트폰 → +수면
BLOCKS = {
    "model 1 (통제: 인구학적+정신건강)"    : DEMO_VARS + MENTAL_VARS,
    "model 2 (+ 스마트폰 의존도/사용시간)" : DEMO_VARS + MENTAL_VARS + SP_VAR,
    "model 3 (+ 수면의 질) = 최종모델"    : DEMO_VARS + MENTAL_VARS + SP_VAR + SL_VAR,
}

fitted       = {}
summary_table = []

for bname, feats in BLOCKS.items():
    X_b      = sm.add_constant(df_sm[feats].astype(float))
    model_sm = sm.GLM(y_sm, X_b, family=sm.families.Binomial(), freq_weights=w_int).fit()
    fitted[bname] = model_sm

    summary_table.append({
        "Block"     : bname,
        "투입변수수": len(feats),
        "Deviance"  : round(float(model_sm.deviance), 2),
        "AIC"       : round(float(model_sm.aic), 2),
        "BIC"       : round(float(model_sm.bic_llf), 2),
    })

    print(f"\n[{bname}]")
    ci_95 = model_sm.conf_int(alpha=0.05)
    ci_95.columns  = ["B_CI_Lower", "B_CI_Upper"]
    or_ci_lower    = np.exp(ci_95["B_CI_Lower"])
    or_ci_upper    = np.exp(ci_95["B_CI_Upper"])

    res_df_sm = pd.DataFrame({
        "B(회귀계수)"    : model_sm.params,
        "S.E.(표준오차)" : model_sm.bse,
        "OR"             : np.exp(model_sm.params),
        "CI(95%)"        : [f"({lo:.4f}, {up:.4f})" for lo, up in zip(or_ci_lower, or_ci_upper)],
        "p-value"        : model_sm.pvalues.apply(format_pvalue),
    }).round(4)
    res_df_sm.index = [VAR_KOR.get(i, i) for i in res_df_sm.index]
    display(res_df_sm)

# [모델 비교] Deviance·AIC·BIC 추이 — 낮을수록 모델 적합도 우수
sum_df = pd.DataFrame(summary_table).set_index("Block")
print("\n▶ 모델 비교 요약표 (Deviance·AIC·BIC 낮을수록 우수)")
display(sum_df)


# =====================================================================
# [시각화] 단계별 모델 적합도 향상 추이
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))
x_labels = [n.replace(" (", "\n(") for n in sum_df.index]
x        = np.arange(len(x_labels))

ax.plot(x, sum_df["Deviance"], marker="o", markersize=10, linewidth=3, label="Deviance", color="#e74c3c")
ax.plot(x, sum_df["AIC"],     marker="s", markersize=10, linewidth=3, linestyle="--", label="AIC", color="#2980b9")
ax.plot(x, sum_df["BIC"],     marker="^", markersize=10, linewidth=3, linestyle=":", label="BIC", color="#27ae60")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
ax.set_ylabel("지수 값 (낮을수록 예측력 우수)", fontsize=13, fontweight="bold")
ax.set_title("위계적 모델링: 단계별 모델 적합도 향상 추이", fontsize=16, fontweight="bold", pad=20)

# [레이블] ax.annotate()로 점 위에 수치 표시 (포인트 단위 오프셋 사용)
for i, val in enumerate(sum_df["Deviance"]):
    ax.annotate(f"{val:,.0f}", (i, val), textcoords="offset points", xytext=(0, 10),
                ha="center", va="bottom", color="#e74c3c", fontweight="bold", fontsize=10)
for i, val in enumerate(sum_df["AIC"]):
    ax.annotate(f"{val:,.0f}", (i, val), textcoords="offset points", xytext=(0, -15),
                ha="center", va="top", color="#2980b9", fontweight="bold", fontsize=10)

ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(fontsize=12, loc="upper right")
plt.tight_layout()
plt.show()
