## 🦷 청소년 스마트폰 의존과 수면의 질에 따른 구강건강 위험도 예측

> **KYRBS 2020 데이터 기반 | XGBoost 머신러닝 | Streamlit 웹 대시보드**

---

## 📌 프로젝트 소개

| 항목 | 내용 |
|------|------|
| **교육과정** | 융합\_데이터 기반 차세대 디지털 헬스케어 AI 솔루션 5기 |
| **주관** | 고용노동부 K-디지털 트레이닝 / 아시아경제 교육센터 |
| **교육 기간** | 2026.02.03 ~ 2026.07.30 |
| **프로젝트 기간** | 2026.03.05 ~ 2026.04.01 |
| **데이터** | 제16차 청소년건강행태조사(KYRBS 2020), 복합표본설계 가중치(W) 적용 |
| **분석 대상** | 청소년 스마트폰 과의존 · 수면의 질 → 구강 건강 위험도 |

청소년의 **스마트폰 과의존**과 **수면의 질**이 구강 건강에 미치는 영향을 분석하고,  
개인 맞춤형 구강 건강 위험도를 예측하는 통합 헬스케어 플랫폼입니다.

단순 질병 예측을 넘어, **복합표본설계 가중치(W)** 를 전면 적용한 정교한 통계 검정을 수행하였으며,  
분석 결과를 바탕으로 개인 맞춤형 행동 개선 가이드와 의료기관 연계 솔루션을 제공합니다.

---

## 🚀 주요 기능

### 1. 정교한 통계 분석 (EDA + 위계적 회귀)
- 복합표본설계 가중치 반영 **Rao-Scott 카이제곱 검정**
- **위계적 다중 로지스틱 회귀분석** (3단계 블록: 인구학적 → 스마트폰 → 수면)
- VIF 기반 다중공선성 검증

### 2. 머신러닝 예측 모델
- **재현율(Recall ≈ 0.85)** 극대화를 목표로 의료 도메인 특성에 맞게 학습
- **Optuna TPE** 하이퍼파라미터 자동 튜닝
- **Youden's Index** 기반 최적 임계값 도출
- 비교 모델: Logistic Regression / Random Forest / XGBoost / LightGBM / CatBoost
- **XGBoost** 최종 선정

### 3. Streamlit 웹 대시보드
- **S-Scale** 스마트폰 과의존 척도 자가진단
- XGBoost 기반 구강 건강 위험도 실시간 예측
- 진단 결과(4그룹)에 따른 1:1 맞춤형 생활습관 가이드
- 주변 치과 검색 및 예약 연계

---

## 💻 실행 방법

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리

```bash
python 01_Data_Preprocessing.py
```

> `data/raw/kyrbs2020.sas7bdat` 원본 데이터 필요  
> 실행 후 `data/processed/kyrbs2020_clean_v1.csv` 생성

### 3. 통계 분석

```bash
python 02_Statistical_Analysis.py
```

### 4. EDA 시각화

```bash
python 03_EDA.py
```

### 5. 머신러닝 학습

```bash
python 04_train.py
```

> 학습 완료 후 `models/` 폴더에 모델·스케일러·메타데이터 저장

### 6. 웹 앱 실행

```bash
streamlit run app.py
```

---

## 🛠 사용 기술

| 분류 | 라이브러리 |
|------|-----------|
| 언어 | Python 3.x |
| 데이터 처리 | pandas, numpy, scipy |
| 통계 분석 | statsmodels |
| 머신러닝 | scikit-learn, XGBoost, LightGBM, CatBoost |
| 하이퍼파라미터 최적화 | Optuna |
| 시각화 | matplotlib, seaborn, plotly |
| 웹 대시보드 | Streamlit |
| 모델 저장 | joblib |

---

## 🔑 주요 구현 포인트

**① 복합표본설계 가중치 적용**  
KYRBS는 층화집락표본설계로 설계된 데이터이므로 단순 분석 시 모집단 대표성이 왜곡됩니다.  
모든 분석(교차분석·회귀·ML)에 가중치(W)를 적용해 모집단 수준의 추정을 수행했습니다.

**② Rao-Scott 카이제곱 검정**  
복합표본 특성을 반영한 설계효과(DEFF) 보정 카이제곱 검정으로  
일반 카이제곱 대비 1종 오류를 줄였습니다.

**③ 의료 도메인 Recall 최적화**  
구강 건강 위험군을 놓치는 비용(False Negative)이 크기 때문에  
목표 Recall ≈ 0.85로 설정하고 임계값을 Youden's Index로 조정했습니다.

**④ 위계적 모델링**  
인구학적 변수 → 정신건강 → 스마트폰 사용 → 수면의 질 순으로  
단계별로 변수를 투입하여 각 블록의 설명력 증분(Deviance·AIC·BIC)을 확인했습니다.

---

## 📁 파일 구조

```
OralHealth_Prediction/
├── app.py                       # Streamlit 웹 애플리케이션 (메인)
├── 01_Data_Preprocessing.py     # 데이터 전처리 및 파생변수 생성
├── 02_Statistical_Analysis.py   # 통계 분석 (카이제곱, 로지스틱 회귀)
├── 03_EDA.py                    # 탐색적 데이터 분석 및 시각화
├── 04_train.py                  # ML 학습 파이프라인
├── requirements.txt             # 패키지 의존성
├── README.md
│
├── data/
│   ├── raw/                     # 원본 SAS 데이터 (KYRBS 2020)
│   └── processed/               # 전처리 완료 CSV
│
├── models/                      # 학습된 모델 저장소
│   ├── xgboost_model.pkl        # 최종 XGBoost 모델
│   ├── scaler.pkl               # 가중치 StandardScaler
│   ├── model_meta.json          # 변수 매핑 및 임계값 정보
│   └── best_params.json         # Optuna 튜닝 결과
│
└── plots/                       # 분석 결과 시각화 이미지
    ├── EDA_01_demographics.png
    ├── EDA_02_smartphone_dependence.png
    └── ...
```

---

## 📊 모델 성능 요약

| 지표 | 값 |
|------|-----|
| 목표 Recall | ≈ 0.85 |
| 최적 임계값 (Youden's Index) | 0.4996 |
| Youden's J | 0.232 |
| Sensitivity | 0.609 |
| Specificity | 0.623 |

> 의료 도메인 특성상 재현율(Recall) 극대화를 우선 목표로 설정

---

## ⚙ 주요 변수

| 변수 유형 | 변수명 |
|----------|--------|
| 인구학적 | 성별, 학교급, 학업성적, 가구소득 |
| 정신건강 | 불안(GAD-7), 스트레스, 절망감, 자살 생각 |
| 스마트폰 | 주중/주말 사용시간, 과의존 여부(S-Scale) |
| 수면 | 수면의 질 (충분/불충분) |
| **타깃** | **구강 건강 문제 여부** (파절·저작불편·통증·잇몸출혈) |
