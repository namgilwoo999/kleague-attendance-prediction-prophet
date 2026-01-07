"""
Facebook Prophet를 활용한 K리그 관중 수 예측
1. Random Forest 피처 중요도 분석
2. Prophet 하이퍼파라미터 그리드 서치
3. 2026 시즌 전체 라운드별 예측

"""

import pandas as pd
import numpy as np
import os
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
 
# 더비 경기 정의 (팀 조합 기반)
# 주요 더비 매치 정의 - 실제 데이터 기반으로 추출
DERBY_PAIRS = {
    # 지역 라이벌
    ('울산', '포항'), ('포항', '울산'),        
    ('서울', '인천'), ('인천', '서울'),        
    ('서울', '수원FC'), ('수원FC', '서울'),    
    ('포항', '대구'), ('대구', '포항'),        
    ('광주', '전북'), ('전북', '광주'),        
    ('대전', '대구'), ('대구', '대전'),          

    # 강팀 라이벌
    ('전북', '울산'), ('울산', '전북'),        
    ('수원FC', '전북'), ('전북', '수원FC'),      
    ('광주', '대구'), ('대구', '광주'),        
}

def is_derby_match(home_team: str, away_team: str) -> int:
    """두 팀의 조합이 더비인지 확인"""
    return 1 if (home_team, away_team) in DERBY_PAIRS else 0

# 1. 데이터 로드 및 피처 엔지니어링
print("\n[1단계] 데이터 로드 및 피처 엔지니어링...")

# 상대 경로
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data/")
results_path = os.path.join(base_dir, "results/")
os.makedirs(results_path, exist_ok=True)

years = [2022, 2023, 2024, 2025]

dfs = []
for year in years:
    try:
        df = pd.read_csv(f"{data_path}k_league1_{year}.csv", encoding='utf-8-sig')
        print(f"  {year}년: {len(df)}건")
        dfs.append(df)
    except Exception as e:
        print(f"  {year}년 실패: {e}")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n총 경기: {len(df_all)}건")

# 기본 전처리
df_all['date'] = pd.to_datetime(df_all['date'])
df_all['attendance'] = pd.to_numeric(df_all['attendance'], errors='coerce')

# 피처 생성
for col in ['temperature', 'humidity', 'is_weekend', 'is_derby']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

df_all['temperature'].fillna(df_all['temperature'].median(), inplace=True)
df_all['humidity'].fillna(df_all['humidity'].median(), inplace=True)
df_all['is_weekend'].fillna(0, inplace=True)
df_all['is_derby'].fillna(0, inplace=True)

# 추가 피처
df_all['month'] = df_all['date'].dt.month
df_all['day_of_week'] = df_all['date'].dt.dayofweek
df_all['is_holiday_season'] = df_all['month'].isin([7, 8, 12, 1]).astype(int)

if 'time' in df_all.columns:
    df_all['hour'] = pd.to_datetime(df_all['time'], format='%H:%M', errors='coerce').dt.hour
    df_all['is_evening'] = (df_all['hour'] >= 18).astype(int)
    df_all['is_evening'].fillna(0, inplace=True)
else:
    df_all['is_evening'] = 0

# 팀별 평균 관중
team_avg = df_all.groupby('home_team')['attendance'].mean().to_dict()
overall_avg = df_all['attendance'].mean()
df_all['home_team_factor'] = df_all['home_team'].map(team_avg) / overall_avg
df_all['home_team_factor'].fillna(1.0, inplace=True)

# 순위 피처
for col in ['home_rank_before', 'away_rank_before']:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

if 'home_rank_before' in df_all.columns and 'away_rank_before' in df_all.columns:
    df_all['rank_avg'] = (df_all['home_rank_before'] + df_all['away_rank_before']) / 2
    df_all['is_top_match'] = ((df_all['home_rank_before'] <= 3) & (df_all['away_rank_before'] <= 3)).astype(int)
    df_all['rank_avg'].fillna(6.5, inplace=True)
else:
    df_all['rank_avg'] = 6.5
    df_all['is_top_match'] = 0

df_all['is_top_match'].fillna(0, inplace=True)

# 라운드 피처
if 'round' in df_all.columns:
    df_all['round'] = pd.to_numeric(df_all['round'], errors='coerce')
    df_all['is_season_start'] = (df_all['round'] <= 5).astype(int)
    df_all['is_season_end'] = (df_all['round'] >= 33).astype(int)
else:
    df_all['is_season_start'] = 0
    df_all['is_season_end'] = 0

df_all['is_season_start'].fillna(0, inplace=True)
df_all['is_season_end'].fillna(0, inplace=True)

# 이상치 제거
df_clean = df_all[df_all['attendance'] >= 500].copy()
print(f"  이상치 제거: {len(df_all) - len(df_clean)}건")

# 2. Random Forest 피처 중요도 분석
print("\n[2단계] Random Forest 피처 중요도 분석...")

feature_cols = ['is_weekend', 'is_derby', 'temperature', 'humidity',
                'home_team_factor', 'rank_avg', 'is_top_match',
                'is_evening', 'is_holiday_season', 'month', 'day_of_week',
                'is_season_start', 'is_season_end']

df_rf = df_clean[feature_cols + ['attendance']].dropna()
X = df_rf[feature_cols]
y = df_rf['attendance']

# Random Forest 학습
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 피처 중요도
importances = pd.DataFrame({
    '피처': feature_cols,
    '중요도': rf.feature_importances_
}).sort_values('중요도', ascending=False)

print("\n  피처 중요도 순위:")
for idx, row in importances.iterrows():
    print(f"    {row['피처']:<20} {row['중요도']:.4f}")

# 상위 피처 선택 (중요도 0.03 이상 또는 상위 8개)
threshold = 0.03
selected_features = importances[importances['중요도'] >= threshold]['피처'].tolist()
if len(selected_features) < 6:
    selected_features = importances.head(8)['피처'].tolist()

print(f"\n  선택된 피처 ({len(selected_features)}개): {', '.join(selected_features)}")

# 3. Prophet 데이터 준비
print("\n[3단계] Prophet 데이터 준비...")

df_prophet = df_clean[['date', 'attendance'] + selected_features].dropna()
df_prophet = df_prophet.rename(columns={'date': 'ds', 'attendance': 'y'})

train_size = int(len(df_prophet) * 0.85)
df_train = df_prophet.iloc[:train_size].copy()
df_test = df_prophet.iloc[train_size:].copy()

print(f"  학습: {len(df_train)}건")
print(f"  검증: {len(df_test)}건")

# 4. 하이퍼파라미터 그리드 서치
print("\n[4단계] 하이퍼파라미터 그리드 서치...")

param_grid = {
    'changepoint_prior_scale': [0.05, 0.1, 0.15],
    'seasonality_prior_scale': [10, 15, 20],
    'seasonality_mode': ['multiplicative']
}

best_params = None
best_score = float('inf')
results = []

print("  그리드 서치 진행 중...")
for cp in param_grid['changepoint_prior_scale']:
    for sp in param_grid['seasonality_prior_scale']:
        for sm in param_grid['seasonality_mode']:
            model = Prophet(
                changepoint_prior_scale=cp,
                seasonality_prior_scale=sp,
                seasonality_mode=sm,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )

            # 회귀변수 추가
            for feat in selected_features:
                model.add_regressor(feat)

            model.fit(df_train)
            forecast = model.predict(df_test)

            mae = mean_absolute_error(df_test['y'], forecast['yhat'])
            results.append({
                'cp': cp,
                'sp': sp,
                'sm': sm,
                'mae': mae
            })

            if mae < best_score:
                best_score = mae
                best_params = {'cp': cp, 'sp': sp, 'sm': sm}

            print(f"    cp={cp}, sp={sp}, sm={sm} -> MAE: {mae:.0f}")

print(f"\n  최적 파라미터:")
print(f"changepoint_prior_scale: {best_params['cp']}")
print(f"seasonality_prior_scale: {best_params['sp']}")
print(f"seasonality_mode: {best_params['sm']}")
print(f"최소 MAE: {best_score:.0f}명")

# 5. 최적 모델 학습
print("\n[5단계] 최적 모델로 재학습...")

final_model = Prophet(
    changepoint_prior_scale=best_params['cp'],
    seasonality_prior_scale=best_params['sp'],
    seasonality_mode=best_params['sm'],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,
    changepoint_range=0.8
)

for feat in selected_features:
    final_model.add_regressor(feat)

final_model.fit(df_train)
print("학습 완료!")

# 6. 성능 평가
print("\n[6단계] 모델 성능 평가...")

forecast_test = final_model.predict(df_test)
actual = df_test['y'].values
predicted = forecast_test['yhat'].clip(lower=0).values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2 = r2_score(actual, predicted)

residuals = actual - predicted

print(f"\n[성능 지표]")
print(f"MAE:{mae:>8,.0f}명")
print(f"RMSE:{rmse:>8,.0f}명")
print(f"MAPE:{mape:>8.2f}%")
print(f"R²:{r2:>8.4f}")

acc_1k = np.mean(np.abs(residuals) <= 1000) * 100
acc_3k = np.mean(np.abs(residuals) <= 3000) * 100
acc_5k = np.mean(np.abs(residuals) <= 5000) * 100

print(f"\n[예측 정확도]")
print(f"±1,000명:{acc_1k:>7.1f}%")
print(f"±3,000명:{acc_3k:>7.1f}%")
print(f"±5,000명:{acc_5k:>7.1f}%")

# 7. 2026 시즌 전체 라운드별 예측
print("\n[7단계] 2026 시즌 전체 라운드별 예측...")

# 2026 일정 CSV 파일이 있는지 확인
schedule_2026_file = os.path.join(data_path, "k_league1_2026_schedule.csv")

if os.path.exists(schedule_2026_file):
    print(f"2026 일정 파일 발견: {schedule_2026_file}")
    print(f"실제 일정 데이터 사용")

    # 실제 일정 로드
    future_2026 = pd.read_csv(schedule_2026_file, encoding='utf-8-sig')
    future_2026['ds'] = pd.to_datetime(future_2026['date'])

    # 더비 경기 자동 감지 (팀 정보가 있는 경우)
    if 'home_team' in future_2026.columns and 'away_team' in future_2026.columns:
        future_2026['is_derby'] = future_2026.apply(
            lambda row: is_derby_match(row['home_team'], row['away_team']),
            axis=1
        )
        print(f"  더비 경기 자동 감지: {future_2026['is_derby'].sum()}경기")
    else:
        future_2026['is_derby'] = 0
        print(f"팀 정보 없음 - 더비 경기 감지 불가")

else:
    print(f"2026 일정 파일 없음")
    print(f"합성 일정 생성 중...")
    print(f"실제 일정이 나오면 '{schedule_2026_file}' 파일을 생성하세요")

    # 합성 스케줄 생성 (38라운드, 주 2경기, 3월 시작)
    season_2026 = []
    start_date = datetime(2026, 3, 1)

    for round_num in range(1, 39):  # 38라운드
        # 라운드 시작 날짜 (대략 주 1회)
        round_date = start_date + timedelta(weeks=round_num-1)

        # 해당 라운드의 경기 생성 (12팀 = 6경기/라운드)
        for game in range(6):
            game_date = round_date + timedelta(days=game % 3)  # 2-3일에 걸쳐 진행

            season_2026.append({
                'ds': game_date,
                'round': round_num,
                'game_in_round': game + 1
            })

    future_2026 = pd.DataFrame(season_2026)

    # 더비 경기 확률적 배정 (역사적 데이터 기반)
    # 전체 경기의 약 15-20%가 더비 경기
    derby_probability = df_clean['is_derby'].mean()
    np.random.seed(42)
    future_2026['is_derby'] = np.random.binomial(1, derby_probability, size=len(future_2026))

    print(f"  합성 일정 생성 완료: {len(future_2026)}경기")
    print(f"  더비 경기 확률적 배정: {future_2026['is_derby'].sum()}경기 (약 {derby_probability*100:.1f}%)")

# 시즌별 평균값 계산
avg_temp = df_train.groupby(df_train['ds'].dt.month)['temperature'].mean().to_dict()
avg_humid = df_train.groupby(df_train['ds'].dt.month)['humidity'].mean().to_dict()

# 피처 생성
future_2026['is_weekend'] = (future_2026['ds'].dt.dayofweek >= 5).astype(int)
future_2026['month'] = future_2026['ds'].dt.month
future_2026['day_of_week'] = future_2026['ds'].dt.dayofweek
future_2026['temperature'] = future_2026['month'].map(avg_temp).fillna(15)
future_2026['humidity'] = future_2026['month'].map(avg_humid).fillna(60)
future_2026['is_evening'] = 1  # 대부분 저녁 경기
future_2026['is_holiday_season'] = future_2026['month'].isin([7, 8, 12, 1]).astype(int)
future_2026['home_team_factor'] = 1.0
future_2026['rank_avg'] = 6.5
future_2026['is_top_match'] = 0

if 'round' in future_2026.columns:
    future_2026['is_season_start'] = (future_2026['round'] <= 5).astype(int)
    future_2026['is_season_end'] = (future_2026['round'] >= 33).astype(int)
else:
    future_2026['is_season_start'] = 0
    future_2026['is_season_end'] = 0

# 예측 실행
forecast_2026 = final_model.predict(future_2026)
forecast_2026['yhat'] = np.maximum(forecast_2026['yhat'].values, 3000)  # 최소 3000명
forecast_2026['yhat_lower'] = np.maximum(forecast_2026['yhat_lower'].values, 0)
forecast_2026['yhat_upper'] = np.maximum(forecast_2026['yhat_upper'].values, 5000)

print(f"  2026 시즌 총 {len(future_2026)}경기 예측 완료")
print(f"  기간: {future_2026['ds'].min().date()} ~ {future_2026['ds'].max().date()}")

# 8. 결과 저장
print("\n[8단계] 결과 저장...")

# 2026 시즌 예측 저장
season_2026_result = pd.DataFrame({
    '날짜': future_2026['ds'].dt.date,
    '요일': future_2026['ds'].dt.day_name(),
    '주말': future_2026['is_weekend'].map({1: '주말', 0: '평일'}),
    '더비': future_2026['is_derby'].map({1: '더비', 0: '일반'}),
    '온도': future_2026['temperature'].round(1),
    '습도': future_2026['humidity'].round(0).astype(int),
    '예측관중': forecast_2026['yhat'].round(0).astype(int),
    '하한95': forecast_2026['yhat_lower'].round(0).astype(int),
    '상한95': forecast_2026['yhat_upper'].round(0).astype(int)
})

# 라운드 정보 추가 (있는 경우)
if 'round' in future_2026.columns:
    season_2026_result.insert(0, '라운드', future_2026['round'])
    season_2026_result.insert(1, '경기번호', future_2026.get('game_in_round', range(1, len(future_2026)+1)))

# 팀 정보 추가 (있는 경우)
if 'home_team' in future_2026.columns and 'away_team' in future_2026.columns:
    season_2026_result.insert(2, '홈팀', future_2026['home_team'])
    season_2026_result.insert(3, '원정팀', future_2026['away_team'])

season_2026_result.to_csv(f"{results_path}2026시즌_전체예측.csv", index=False, encoding='utf-8-sig')
print("2026시즌_전체예측.csv 저장")

# 라운드별 요약 (라운드 정보가 있는 경우)
if 'round' in future_2026.columns:
    round_summary = season_2026_result.groupby('라운드').agg({
        '날짜': 'first',
        '예측관중': ['mean', 'min', 'max'],
        '더비': lambda x: (x == '더비').sum()
    })
    round_summary.columns = ['시작날짜', '평균관중', '최소관중', '최대관중', '더비경기수']
    for col in ['평균관중', '최소관중', '최대관중', '더비경기수']:
        round_summary[col] = round_summary[col].astype(int)
    round_summary.to_csv(f"{results_path}2026시즌_라운드별요약.csv", encoding='utf-8-sig')
    print("2026시즌_라운드별요약.csv 저장")

# 검증 데이터 결과 저장
validation_result = pd.DataFrame({
    '날짜': df_test['ds'],
    '실제관중': actual,
    '예측관중': predicted.astype(int),
    '오차': (actual - predicted).astype(int),
    '하한95': np.maximum(forecast_test['yhat_lower'].values, 0).astype(int),
    '상한95': forecast_test['yhat_upper'].values.astype(int)
})
validation_result.to_csv(f"{results_path}검증데이터_예측결과.csv", index=False, encoding='utf-8-sig')
print("검증데이터_예측결과.csv 저장")

# 피처 중요도 저장
importances.to_csv(f"{results_path}피처중요도.csv", index=False, encoding='utf-8-sig')
print("피처중요도.csv 저장")

# 성능 지표 저장
performance = pd.DataFrame({
    '지표': ['MAE', 'RMSE', 'MAPE', 'R²', '±1000명', '±3000명', '±5000명'],
    '값': [mae, rmse, mape, r2, acc_1k, acc_3k, acc_5k],
    '단위': ['명', '명', '%', '-', '%', '%', '%']
})
performance.to_csv(f"{results_path}모델성능.csv", index=False, encoding='utf-8-sig')
print("모델성능.csv 저장")

# 최적 파라미터 저장
params_df = pd.DataFrame([best_params])
params_df.to_csv(f"{results_path}최적파라미터.csv", index=False, encoding='utf-8-sig')
print(" 최적파라미터.csv 저장")

# 더비 팀 조합 저장
derby_df = pd.DataFrame(list(DERBY_PAIRS), columns=['홈팀', '원정팀'])
derby_df.to_csv(f"{results_path}더비_팀조합.csv", index=False, encoding='utf-8-sig')
print("더비_팀조합.csv 저장")

# 9. 최종 요약
print("\n" + "=" * 80)
print("최종 결과 요약")
print("=" * 80)

print(f"\n[1] 피처 선택")
print(f"Random Forest로 선택된 피처: {len(selected_features)}개")
print(f"{', '.join(selected_features)}")

print(f"\n[2] 더비 경기 정의")
print(f"정의된 더비 조합: {len(DERBY_PAIRS) // 2}개 (양방향)")
print(f"예: 울산 vs 포항, 서울 vs 인천, 전북 vs 울산 등")

print(f"\n[3] 최적 하이퍼파라미터")
print(f"changepoint_prior_scale: {best_params['cp']}")
print(f"seasonality_prior_scale: {best_params['sp']}")
print(f"seasonality_mode: {best_params['sm']}")

print(f"\n[4] 모델 성능 (검증 데이터)")
print(f"MAE:{mae:>8,.0f}명")
print(f"RMSE:{rmse:>8,.0f}명")
print(f"MAPE:{mape:>8.2f}%")
print(f"R²:{r2:>8.4f}")

print(f"\n[5] 예측 정확도")
print(f"±1,000명:{acc_1k:>7.1f}%")
print(f"±3,000명:{acc_3k:>7.1f}%")
print(f"±5,000명:{acc_5k:>7.1f}%")

print(f"\n[6] 2026 시즌 예측")
print(f"총 경기:{len(season_2026_result):>7}경기")
if 'round' in future_2026.columns:
    print(f"라운드:{season_2026_result['라운드'].max():>7}라운드")
print(f"평균 관중:{season_2026_result['예측관중'].mean():>7,.0f}명")
print(f"최소 관중:{season_2026_result['예측관중'].min():>7,.0f}명")
print(f"최대 관중:{season_2026_result['예측관중'].max():>7,.0f}명")

derby_avg = season_2026_result[season_2026_result['더비'] == '더비']['예측관중'].mean()
normal_avg = season_2026_result[season_2026_result['더비'] == '일반']['예측관중'].mean()

print(f"\n[7] 더비 효과")
print(f"더비 평균:{derby_avg:>7,.0f}명")
print(f"일반 평균:{normal_avg:>7,.0f}명")
print(f"효과:{((derby_avg/normal_avg - 1) * 100):>7.1f}%")

print(f"\n[8] 월별 예측 (2026)")
monthly_summary = season_2026_result.groupby(season_2026_result['날짜'].apply(lambda x: pd.to_datetime(x).month))['예측관중'].mean()
for month, avg in monthly_summary.items():
    print(f"{month}월: {avg:>7,.0f}명")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)

print("\n생성된 파일:")
print("[데이터]")
print("2026시즌_전체예측.csv - 전체 경기 예측")
if 'round' in future_2026.columns:
    print("2026시즌_라운드별요약.csv - 라운드 요약")
print("검증데이터_예측결과.csv - 검증 성능")
print("피처중요도.csv - 피처 중요도")
print("모델성능.csv - 성능 지표")
print("최적파라미터.csv - 최적 파라미터")
print("더비_팀조합.csv - 더비 팀 조합 정의")
print("\n[시각화]")
print("시각화는 visualization.ipynb를 실행하세요")
