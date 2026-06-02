"""
[목적] KYRBS 2020 원본 SAS 데이터를 불러와 분석에 필요한 파생변수를 생성하고
       결측치를 제거한 뒤 전처리 완료 CSV로 저장합니다.
[입력] data/raw/kyrbs2020.sas7bdat
[출력] data/processed/kyrbs2020_clean_v1.csv
"""

import pandas as pd
import numpy as np

# =====================================================================
# [로드] 원본 SAS 데이터에서 분석 대상 변수만 추출
# =====================================================================
df = pd.read_sas('./data/raw/kyrbs2020.sas7bdat', format='sas7bdat')

TARGET_COLS = [
    'SEX', 'GRADE', 'E_S_RCRD', 'E_SES',
    'INT_SPWD_TM', 'INT_SPWK_TM',
    'INT_SP_OU_1', 'INT_SP_OU_2', 'INT_SP_OU_3', 'INT_SP_OU_4', 'INT_SP_OU_5',
    'INT_SP_OU_6', 'INT_SP_OU_7', 'INT_SP_OU_8', 'INT_SP_OU_9', 'INT_SP_OU_10',
    'M_GAD_1', 'M_GAD_2', 'M_GAD_3', 'M_GAD_4', 'M_GAD_5', 'M_GAD_6', 'M_GAD_7',
    'M_STR', 'M_SAD', 'M_SUI_CON',
    'O_SYMP1', 'O_SYMP2', 'O_SYMP3', 'O_SYMP4',
    'M_SLP_EN', 'W'
]
df = df[TARGET_COLS].copy()


# =====================================================================
# [파생변수] 인구학적 변수 — 성별 · 학교급 · 성적 · 경제수준
# =====================================================================

# [성별] 남→여 순으로 범주형 정의
df['gender'] = pd.Categorical(
    df['SEX'].map({1: 'Male', 2: 'Female'}),
    categories=['Male', 'Female'], ordered=True
)

# [학교급] 1~3학년=중학교, 4~6학년=고등학교
df['school'] = df['GRADE'].apply(
    lambda x: 'Middle school' if x in [1, 2, 3] else ('High school' if x in [4, 5, 6] else np.nan)
)
df['school'] = pd.Categorical(df['school'], categories=['Middle school', 'High school'], ordered=True)

# [성적 · 경제수준] 상/중/하 3단계 범주화 (1~2=상, 3=중, 4~5=하)
_level_map = {1: 'High', 2: 'High', 3: 'Middle', 4: 'Low', 5: 'Low'}
_level_cats = ['High', 'Middle', 'Low']
df['grade']  = pd.Categorical(df['E_S_RCRD'].map(_level_map), categories=_level_cats, ordered=True)
df['income'] = pd.Categorical(df['E_SES'].map(_level_map),    categories=_level_cats, ordered=True)


# =====================================================================
# [파생변수] 스마트폰 사용 시간 범주화 (분 → 시간 → 구간)
# =====================================================================
def categorize_time(minutes: float) -> str:
    """
    [역할] 분(minutes) 단위 스마트폰 사용시간을 4구간으로 범주화합니다.
    [반환] '≤3' | '3 ~ 5' | '5 ~ 8' | '≥8' | np.nan
    """
    if pd.isna(minutes):
        return np.nan
    hours = minutes / 60
    if hours <= 3:       return '≤3'
    elif hours <= 5:     return '3 ~ 5'
    elif hours <= 8:     return '5 ~ 8'
    else:                return '≥8'

TIME_CATS = ['≤3', '3 ~ 5', '5 ~ 8', '≥8']
df['smartphone_use_day']     = pd.Categorical(df['INT_SPWD_TM'].apply(categorize_time), categories=TIME_CATS, ordered=True)
df['smartphone_use_weekend'] = pd.Categorical(df['INT_SPWK_TM'].apply(categorize_time), categories=TIME_CATS, ordered=True)


# =====================================================================
# [파생변수] 스마트폰 의존도 — S-Scale 10문항 합산
# =====================================================================
# [기준] 합산 점수 23점 미만=일반군(No), 23점 이상=위험군(Risk)
SP_COLS = [f'INT_SP_OU_{i}' for i in range(1, 11)]
df['sp_score'] = df[SP_COLS].sum(axis=1)
df['smartphone_dependence'] = pd.Categorical(
    df['sp_score'].apply(lambda x: 'No' if x < 23 else 'Risk'),
    categories=['No', 'Risk'], ordered=True
)


# =====================================================================
# [파생변수] 불안 — GAD-7 문항 합산 후 4단계 분류
# =====================================================================
# [전처리] 각 문항에서 1을 빼서 0-기반 점수로 변환 후 합산
GAD_COLS = [f'M_GAD_{i}' for i in range(1, 8)]
df['anxiety_score'] = df[GAD_COLS].apply(lambda x: x - 1).sum(axis=1)

def categorize_anxiety(score: float) -> str:
    """
    [역할] GAD-7 합산 점수를 불안 단계로 분류합니다.
    [기준] ≤4=없음 / ≤9=가벼움 / ≤14=중간 / 이상=심각
    """
    if pd.isna(score): return np.nan
    if score <= 4:     return 'No'
    elif score <= 9:   return 'Mild'
    elif score <= 14:  return 'Moderate'
    else:              return 'Severe'

df['anxiety'] = pd.Categorical(
    df['anxiety_score'].apply(categorize_anxiety),
    categories=['No', 'Mild', 'Moderate', 'Severe'], ordered=True
)


# =====================================================================
# [파생변수] 정신건강 — 스트레스 · 절망감 · 자살 생각
# =====================================================================
df['stress'] = pd.Categorical(
    df['M_STR'].map({1: 'High', 2: 'High', 3: 'Middle', 4: 'Low', 5: 'Low'}),
    categories=['High', 'Middle', 'Low'], ordered=True
)
df['despair']           = pd.Categorical(df['M_SAD'].map({1: 'No', 2: 'Yes'}),     categories=['No', 'Yes'], ordered=True)
df['suicidal_thoughts'] = pd.Categorical(df['M_SUI_CON'].map({1: 'No', 2: 'Yes'}), categories=['No', 'Yes'], ordered=True)


# =====================================================================
# [파생변수] 구강 건강 — 4가지 증상 유무
# =====================================================================
# [개별 증상] 치아 파절 / 저작 불편 / 치아 통증 / 잇몸 출혈
df['tooth_fracture']    = pd.Categorical(df['O_SYMP1'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['chewing_discomfort']= pd.Categorical(df['O_SYMP2'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['tooth_pain']        = pd.Categorical(df['O_SYMP3'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)
df['gingival_bleeding'] = pd.Categorical(df['O_SYMP4'].map({0: 'No', 1: 'Yes'}), categories=['No', 'Yes'], ordered=True)

# [통합 지표] 4개 증상 합산 점수 (0=없음, 1~4=증상 있음)
df['oral_health_score'] = df[['O_SYMP1', 'O_SYMP2', 'O_SYMP3', 'O_SYMP4']].sum(axis=1)


# =====================================================================
# [파생변수] 수면의 질 — 충분(No 문제) vs 부족(Yes 문제)
# =====================================================================
# [기준] 응답 1~2=충분(No), 3~5=불충분(Yes)
df['sleep_quality'] = pd.Categorical(
    df['M_SLP_EN'].map({1: 'No', 2: 'No', 3: 'Yes', 4: 'Yes', 5: 'Yes'}),
    categories=['No', 'Yes'], ordered=True
)


# =====================================================================
# [최종 데이터셋] 분석 대상 변수만 선택 후 결측치 제거
# =====================================================================
FINAL_COLS = [
    'gender', 'school', 'grade', 'income',
    'smartphone_use_day', 'smartphone_use_weekend',
    'smartphone_dependence',
    'anxiety', 'stress', 'despair', 'suicidal_thoughts',
    'tooth_fracture', 'chewing_discomfort', 'tooth_pain', 'gingival_bleeding',
    'oral_health_score',
    'sleep_quality',
    'W'   # [가중치] 복합표본설계 가중치 — 모든 분석에 사용
]
df_final = df[FINAL_COLS]

# [확인] 결측치 현황 출력
print("[결측치 현황]")
print(df_final.isnull().sum())

# [결측치 제거] listwise deletion 적용
df_clean = df_final.dropna().copy()
print(f"\n제거 전 데이터 수: {len(df_final)}")
print(f"제거 후 데이터 수: {len(df_clean)}")   # 선행 논문과 동일한 50,975건 예상

# [저장] 전처리 완료 데이터 CSV로 저장
df_clean.to_csv('./data/processed/kyrbs2020_clean_v1.csv', index=False, encoding='utf-8-sig')
print("\n[완료] ./data/processed/kyrbs2020_clean_v1.csv 저장 완료")
