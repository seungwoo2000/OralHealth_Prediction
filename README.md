<!--
████████████████████████████████████████████████████████████████████
  seungwoo2000 · OralHealth_Prediction — K-디지털 트레이닝 포트폴리오 README
████████████████████████████████████████████████████████████████████
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0c4a6e,50:0284c7,100:38bdf8&height=220&section=header&text=🦷%20구강건강%20위험도%20예측&fontSize=46&fontColor=ffffff&fontAlignY=40&desc=청소년%20스마트폰%20의존%20×%20수면의%20질%20→%20AI%20예측&descAlignY=62&descColor=bae6fd&animation=fadeIn" alt="header" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-최종모델-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-대시보드-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-하이퍼파라미터-6C5CE7?style=for-the-badge)
![KYRBS](https://img.shields.io/badge/KYRBS_2020-공공데이터-00B894?style=for-the-badge)

<br/>

> **스마트폰을 많이 쓰고 잠을 못 자는 청소년, 치아도 위험하다?**  
> KYRBS 2020 공공데이터 기반 · XGBoost 머신러닝 · Streamlit 웹 대시보드

</div>

---

## 📋 훈련 과정 정보

| 항목 | 내용 |
|:---|:---|
| 🏫 **훈련기관** | 아시아경제 교육센터 |
| 📚 **훈련과정명** | 융합\_데이터 기반 차세대 디지털 헬스케어 AI 솔루션 5회차 |
| 🏷️ **훈련유형** | K-디지털 트레이닝 (고용노동부) |
| 📅 **훈련기간** | 2026-02-03 ~ 2026-07-30 (6개월) |
| 🔬 **프로젝트 기간** | 2026-03-05 ~ 2026-04-01 |
| 💡 **프로젝트 분류** | 헬스케어 AI · 머신러닝 예측 모델 · 통계 분석 |

---

## 🦷 프로젝트 소개

```
"스마트폰을 오래 쓰면 잠도 못 자고, 잠을 못 자면 치아도 나빠진다?"

이 가설을 청소년 62만 명 데이터로 검증하고,
AI로 개인 맞춤 구강 건강 위험도를 예측합니다.
```

**청소년의 스마트폰 과의존**과 **수면의 질**이 구강 건강에 미치는 영향을 분석하고,  
개인 맞춤형 구강 건강 위험도를 예측하는 **통합 헬스케어 AI 플랫폼**입니다.

| 🔑 키워드 | 설명 |
|:---|:---|
| **KYRBS 2020** | 국가가 수집한 청소년 건강 행태 조사 공공데이터 (제16차) |
| **복합표본 가중치(W)** | 표본이 전체 청소년을 대표하도록 보정하는 통계 기법 |
| **XGBoost** | 높은 정확도로 유명한 그래디언트 부스팅 머신러닝 모델 |
| **Recall 최적화** | 위험군을 놓치지 않는 것을 최우선으로 설계한 의료 AI |

---

## 🗂️ 목차

1. [분석 파이프라인](#-분석-파이프라인)
2. [주요 기능](#-주요-기능)
3. [모델 성능](#-모델-성능)
4. [주요 변수](#-주요-변수)
5. [기술 스택](#-기술-스택)
6. [파일 구조](#-파일-구조)
7. [핵심 구현 포인트](#-핵심-구현-포인트)
8. [배운 점 · 성장 포인트](#-배운-점--성장-포인트)
9. [실행 방법](#-실행-방법)

---

## 🔄 분석 파이프라인

```
① 공공데이터 수집        KYRBS 2020 (SAS 형식 원본)
        ↓
② 데이터 전처리          결측값 처리 · 파생변수 생성 · 가중치 적용
        ↓
③ 통계 분석 (EDA)        Rao-Scott 카이제곱 · 위계적 로지스틱 회귀
        ↓
④ 머신러닝 학습          5개 모델 비교 → XGBoost 최종 선정
        ↓
⑤ 하이퍼파라미터 튜닝    Optuna TPE 자동 최적화
        ↓
⑥ 임계값 최적화          Youden's Index 기반 최적 임계값 도출
        ↓
⑦ 웹 대시보드 배포       Streamlit 실시간 예측 + 맞춤 가이드
```

---

## 🚀 주요 기능

### 1️⃣ 정교한 통계 분석

```
📊  Rao-Scott 카이제곱    →  복합표본 특성을 반영한 오차 보정 검정
📈  위계적 로지스틱 회귀   →  변수를 3단계로 나눠 단계별 영향력 측정
🔍  VIF 다중공선성 검증    →  변수 간 중복 정보 여부 사전 확인
```

> 💡 **위계적 회귀란?** 변수를 한꺼번에 넣는 게 아니라  
> 인구학적 요인 → 스마트폰 → 수면 순으로 단계적으로 투입해  
> 각 요인이 추가로 설명하는 양을 측정하는 방법입니다.

### 2️⃣ 머신러닝 예측 모델

| 비교 모델 | 선정 여부 |
|:---:|:---:|
| Logistic Regression | — |
| Random Forest | — |
| **XGBoost** | ✅ **최종 선정** |
| LightGBM | — |
| CatBoost | — |

```
🎯  목표 Recall ≈ 0.85    →  위험군을 놓치지 않는 것을 최우선
⚙️  Optuna TPE 튜닝       →  수백 번 자동 실험으로 최적 파라미터 탐색
📏  Youden's Index        →  민감도+특이도를 동시에 최대화하는 임계값 설정
```

### 3️⃣ Streamlit 웹 대시보드

```
📱  S-Scale 자가진단      →  스마트폰 과의존 척도 직접 입력
🤖  실시간 위험도 예측     →  XGBoost가 즉시 구강 건강 위험도 계산
📋  4그룹 맞춤 가이드      →  진단 결과에 따른 1:1 생활습관 개선 안내
🏥  치과 연계 서비스       →  주변 치과 검색 및 예약 연결
```

---

## 📊 모델 성능

| 지표 | 값 | 의미 |
|:---:|:---:|:---|
| **목표 Recall** | ≈ 0.85 | 위험군 중 85%를 놓치지 않고 탐지 |
| **최적 임계값** | 0.4996 | Youden's Index로 도출한 예측 기준선 |
| **Youden's J** | 0.232 | 민감도 + 특이도의 균형 지수 |
| **Sensitivity** | 0.609 | 실제 위험군을 위험으로 예측한 비율 |
| **Specificity** | 0.623 | 실제 정상군을 정상으로 예측한 비율 |

> ⚕️ 의료 도메인 특성상 **재현율(Recall) 극대화**를 최우선으로 설계  
> 위험군을 놓치는 비용(False Negative)이 크기 때문입니다.

---

## ⚙️ 주요 변수

| 변수 유형 | 변수명 |
|:---|:---|
| 👤 **인구학적** | 성별, 학교급, 학업성적, 가구소득 |
| 🧠 **정신건강** | 불안(GAD-7), 스트레스, 절망감, 자살 생각 |
| 📱 **스마트폰** | 주중/주말 사용시간, 과의존 여부(S-Scale) |
| 😴 **수면** | 수면의 질 (충분 / 불충분) |
| 🦷 **타깃 (예측 대상)** | 구강 건강 문제 여부 (파절·저작불편·통증·잇몸출혈) |

---

## 🛠️ 기술 스택

> 비전공자도 이해할 수 있도록, 각 기술이 **어떤 역할**을 하는지 함께 설명합니다.

| 분류 | 기술 | 한 줄 설명 |
|:---:|:---:|:---|
| **언어** | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | 데이터 분석과 AI 개발의 표준 언어 |
| **데이터 처리** | `pandas` `numpy` `scipy` | 데이터를 불러오고 가공하는 도구 |
| **통계 분석** | `statsmodels` | 회귀분석·가설검정 등 통계 전용 라이브러리 |
| **머신러닝** | `scikit-learn` `XGBoost` `LightGBM` `CatBoost` | 예측 모델 학습 |
| **하이퍼파라미터** | `Optuna` | 모델 설정값을 자동으로 최적화하는 도구 |
| **시각화** | `matplotlib` `seaborn` `plotly` | 분석 결과를 그래프로 표현 |
| **웹 대시보드** | `Streamlit` | Python 코드만으로 웹 앱을 만드는 프레임워크 |
| **모델 저장** | `joblib` | 학습된 AI 모델을 파일로 저장·불러오기 |

---

## 📁 파일 구조

```
OralHealth_Prediction/
│
├── app.py                       # 🌐 Streamlit 웹 대시보드 (메인)
├── 01_Data_Preprocessing.py     # 데이터 전처리 · 파생변수 생성
├── 02_Statistical_Analysis.py   # 통계 분석 (카이제곱 · 로지스틱 회귀)
├── 03_EDA.py                    # 탐색적 데이터 분석 · 시각화
├── 04_train.py                  # ML 학습 파이프라인
├── requirements.txt             # 패키지 의존성 목록
│
├── 📂 data/
│   ├── raw/                     # 원본 SAS 데이터 (KYRBS 2020)
│   └── processed/               # 전처리 완료 CSV
│
├── 📂 models/
│   ├── xgboost_model.pkl        # 최종 XGBoost 모델
│   ├── scaler.pkl               # 가중치 StandardScaler
│   ├── model_meta.json          # 변수 매핑 · 임계값 정보
│   └── best_params.json         # Optuna 튜닝 결과
│
└── 📂 plots/                    # 분석 결과 시각화 이미지
    ├── EDA_01_demographics.png
    ├── EDA_02_smartphone_dependence.png
    └── ...
```

---

## 🔑 핵심 구현 포인트

**① 복합표본설계 가중치(W) 전면 적용**
> KYRBS는 층화집락표본설계로 수집된 데이터라 가중치 없이 분석하면  
> 결과가 실제 청소년 전체를 제대로 반영하지 못합니다.  
> 교차분석·회귀·ML 모든 단계에 가중치를 적용해 모집단 수준 추정을 수행했습니다.

**② Rao-Scott 카이제곱 검정**
> 복합표본의 특성인 설계효과(DEFF)를 보정한 카이제곱 검정으로,  
> 일반 카이제곱 대비 1종 오류(잘못된 유의미 판정)를 줄였습니다.

**③ 의료 도메인 Recall 최적화**
> 위험군을 정상으로 잘못 분류하는 비용(False Negative)이 크기 때문에  
> 목표 Recall ≈ 0.85로 설정하고 Youden's Index로 임계값을 조정했습니다.

**④ 위계적 블록 모델링**
> 인구학적 변수 → 정신건강 → 스마트폰 → 수면 순으로 단계 투입,  
> 각 블록이 추가적으로 설명하는 예측력 증분을 AIC·BIC로 검증했습니다.

---

## 📈 배운 점 · 성장 포인트

| 분야 | 배운 것 | 이걸 배워서 뭘 할 수 있게 됐나? |
|:---|:---|:---|
| 📊 **공공데이터 분석** | KYRBS SAS 데이터 로드 · 가중치 처리 | 국가 통계 데이터를 실제 분석에 활용 |
| 🔬 **통계 검정** | Rao-Scott 카이제곱 · 위계적 회귀 | 데이터 특성에 맞는 통계 기법 선택 |
| 🤖 **머신러닝** | 5개 모델 비교 → XGBoost 선정 | 문제에 맞는 모델을 실험으로 찾는 과정 |
| ⚙️ **AutoML** | Optuna TPE 하이퍼파라미터 튜닝 | 수동 튜닝 없이 자동으로 최적 설정 탐색 |
| ⚕️ **의료 AI 설계** | Recall 최적화 · Youden's Index | 도메인 특성에 맞는 평가지표 설정 방법 |
| 🌐 **웹 배포** | Streamlit 대시보드 구축 | 분석 결과를 누구나 쓸 수 있는 서비스로 전환 |
| 🏥 **헬스케어 기획** | 4그룹 맞춤 가이드 설계 | AI 예측 결과를 실질적 행동 변화로 연결 |

---

## ⚙️ 실행 방법

**1️⃣ 환경 설치**
```bash
pip install -r requirements.txt
```

**2️⃣ 데이터 전처리**
```bash
python 01_Data_Preprocessing.py
```
> `data/raw/kyrbs2020.sas7bdat` 원본 데이터 필요  
> 실행 후 `data/processed/kyrbs2020_clean_v1.csv` 생성

**3️⃣ 통계 분석**
```bash
python 02_Statistical_Analysis.py
```

**4️⃣ EDA 시각화**
```bash
python 03_EDA.py
```

**5️⃣ 머신러닝 학습**
```bash
python 04_train.py
```
> 학습 완료 후 `models/` 폴더에 모델·스케일러·메타데이터 저장

**6️⃣ 웹 대시보드 실행**
```bash
streamlit run app.py
```

---

<div align="center">

<br/>

*"데이터로 증명하고, AI로 예측하고, 서비스로 연결한다."* 🦷

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:38bdf8,50:0284c7,100:0c4a6e&height=130&section=footer&text=K-디지털%20트레이닝%20|%20아시아경제%20교육센터&fontSize=15&fontColor=ffffff&fontAlignY=65" width="100%"/>

**📅 2026.03.05 ~ 2026.04.01** &nbsp;|&nbsp; Made with 🦷 during K-Digital Training

</div>
