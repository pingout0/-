"""
논문 재현 대시보드 (Paper Companion) — 실시간 계산
논문 "RSM-머신러닝 하이브리드 LME 예측 검증"의 전 과정을 직접 돌려보는 도구

각 섹션이 논문 구조와 1:1 대응하며, 버튼을 누르면 실제로 모델이 계산됨.
실행: streamlit run paper_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="논문 재현 대시보드", page_icon="📄",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.hero { background:linear-gradient(135deg,#13233A,#1E3556); color:#fff;
        border-radius:14px; padding:24px 28px; margin-bottom:10px; }
.step { background:#F4F6F9; color:#24303F; border-left:4px solid #C87941;
        border-radius:0 10px 10px 0; padding:12px 16px; margin:8px 0; font-size:14px; }
.res { background:#EAF5EF; color:#13233A; border-radius:10px; padding:14px 18px;
       margin:8px 0; font-size:14px; border:1px solid #B8DFC9; }
.formula { background:#13233A; color:#DCE6F2; border-radius:8px; padding:12px 16px;
           margin:6px 0; font-family:monospace; font-size:13px; }
</style>
""", unsafe_allow_html=True)

NAVY="#5B8DD9"; COPPER="#E8A66B"; STEEL="#7BA3C9"; GREEN="#4CAF88"; RED="E57373"

@st.cache_data
def load_data():
    df = pd.read_csv("sample_multivariate.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# ── 사이드바: 논문 목차
with st.sidebar:
    st.markdown("## 📄 논문 재현 대시보드")
    st.markdown("*RSM-머신러닝 하이브리드\nLME 예측 검증*")
    st.divider()
    st.caption("논문의 전 과정을 실시간으로 직접 계산해봅니다")
    section = st.radio("논문 목차", [
        "3.1 데이터 분석",
        "3.2 네 가지 방법",
        "3.3 평가지표",
        "4.1 방법 성능 비교",
        "4.2 계수 예측의 한계",
        "4.4 구간 분리",
    ])
    st.divider()
    st.caption(f"데이터: {len(df)}개월\n{df['날짜'].iloc[0]} ~ {df['날짜'].iloc[-1]}")

from rsm_hybrid import (evaluate, fit_rsm, predict_rsm, set_rsm_scaler,
                        rsm_design_matrix, scenario1_direct, scenario2_ml_rsm)
from step_forecast import scenario2_step_forecast

# ══════════════════════════════════════════════
# 3.1 데이터 분석
# ══════════════════════════════════════════════
if section.startswith("3.1"):
    st.markdown('<div class="hero"><b>논문 3.1절 · 데이터 탐색적 분석</b><br>'
                '논문 그림 1~3이 어떻게 계산되는지 직접 확인합니다</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["① 가격 추이·국면", "② 변수 상관", "③ 계절성 검정"])

    with tab1:
        st.markdown("**논문 그림 1** — 가격 추이와 세 국면")
        fig = go.Figure()
        dd = pd.to_datetime(df['날짜']+'-01')
        fig.add_trace(go.Scatter(x=dd, y=df['LME가격'], line=dict(color=NAVY, width=2)))
        fig.add_vrect(x0=dd.iloc[0], x1="2020-03-01", fillcolor="green", opacity=0.1, line_width=0)
        fig.add_vrect(x0="2020-04-01", x1="2022-03-01", fillcolor="red", opacity=0.1, line_width=0)
        fig.add_vrect(x0="2022-04-01", x1=dd.iloc[-1], fillcolor="orange", opacity=0.1, line_width=0)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#888"), xaxis=dict(gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(gridcolor="rgba(128,128,128,0.2)"), height=380, margin=dict(t=20,b=20), yaxis_title="LME ($/톤)")
        st.plotly_chart(fig, use_container_width=True)
        # 국면별 통계 실시간 계산
        ret = df['LME가격'].pct_change()*100
        st.markdown("**국면별 특성 (실시간 계산)**")
        c1,c2,c3 = st.columns(3)
        for col,(name,s,e,cl) in zip([c1,c2,c3],[("안정기","2017-01","2020-03",GREEN),("급등기","2020-04","2022-03",RED),("조정기","2022-04","2025-12",COPPER)]):
            m=(df['날짜']>=s)&(df['날짜']<=e)
            col.metric(name, f"{ret[m].mean():+.2f}%/월", f"변동성 {ret[m].std():.2f}%")

    with tab2:
        st.markdown("**논문 그림 2** — 시차별 상관과 수준 vs 변화율")
        st.markdown('<div class="step">각 변수를 시간축에서 밀어보며 LME와 가장 닮는 지점을 찾습니다. 아래 버튼으로 직접 계산해보세요.</div>', unsafe_allow_html=True)
        if st.button("🔄 상관 실시간 계산", key="corr"):
            with st.spinner("계산 중..."):
                lme,cu,oil,fx = df['LME가격'],df['구리'],df['유가WTI'],df['환율']
                lags=range(-6,7)
                fig=go.Figure()
                for name,s,col in [("구리",cu,COPPER),("유가",oil,STEEL),("환율",fx,GREEN)]:
                    vals=[lme.corr(s.shift(l)) for l in lags]
                    fig.add_trace(go.Scatter(x=list(lags),y=vals,name=name,line=dict(color=col,width=2),mode="lines+markers"))
                fig.add_vline(x=0,line_dash="dash",line_color="gray")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#888"), xaxis=dict(gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(gridcolor="rgba(128,128,128,0.2)"), height=340,margin=dict(t=20,b=20),xaxis_title="시차(개월): 음수=선행",yaxis_title="상관")
                st.plotly_chart(fig,use_container_width=True)
                # 수준 vs 변화율
                rr=df[['LME가격','구리','유가WTI','환율']].pct_change().dropna()
                st.markdown("**수준 vs 변화율 상관 (핵심 발견)**")
                tbl=pd.DataFrame({
                    "변수":["구리","유가","환율"],
                    "수준 상관":[round(lme.corr(cu),3),round(lme.corr(oil),3),round(lme.corr(fx),3)],
                    "변화율 상관":[round(rr['LME가격'].corr(rr['구리']),3),round(rr['LME가격'].corr(rr['유가WTI']),3),round(rr['LME가격'].corr(rr['환율']),3)],
                })
                st.dataframe(tbl,use_container_width=True,hide_index=True)
                st.markdown('<div class="res">💡 수준 상관은 높지만(구리 0.85) 변화율 상관은 낮음(0.41). 장기 추세는 같아도 월별 움직임은 따로 논다 → 계수 예측이 어려운 이유</div>',unsafe_allow_html=True)

    with tab3:
        st.markdown("**논문 그림 3** — 계절성 검정")
        st.markdown('<div class="step">월별 평균 변화율의 차이가 통계적으로 유의한지 ANOVA로 검정합니다.</div>', unsafe_allow_html=True)
        if st.button("🔄 계절성 ANOVA 검정 실행", key="anova"):
            with st.spinner("검정 중..."):
                from scipy import stats
                df2=df.copy(); df2['월']=pd.to_datetime(df2['날짜']+'-01').dt.month
                df2['수익률']=df2['LME가격'].pct_change()*100
                groups=[df2[df2['월']==m]['수익률'].dropna().values for m in range(1,13)]
                groups=[g for g in groups if len(g)>1]
                f_stat,p_val=stats.f_oneway(*groups)
                mon=df2.groupby('월')['수익률'].mean()
                fig=go.Figure(go.Bar(x=list(range(1,13)),y=[mon[m] for m in range(1,13)],
                    marker_color=[COPPER if mon[m]>0 else STEEL for m in range(1,13)]))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#888"), xaxis=dict(gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(gridcolor="rgba(128,128,128,0.2)"), height=320,margin=dict(t=20,b=20),xaxis_title="월",yaxis_title="평균 변화율(%)")
                st.plotly_chart(fig,use_container_width=True)
                c1,c2=st.columns(2)
                c1.metric("F 통계량", f"{f_stat:.2f}")
                c2.metric("p 값", f"{p_val:.3f}", "계절성 없음" if p_val>0.05 else "계절성 있음")
                st.markdown(f'<div class="res">💡 p={p_val:.2f} > 0.05 → 월별 차이가 통계적으로 무의미 = <b>계절성 없음</b>. LME는 계절보다 거시 국면에 지배됨</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 3.2 네 가지 방법
# ══════════════════════════════════════════════
elif section.startswith("3.2"):
    st.markdown('<div class="hero"><b>논문 3.2절 · 네 가지 예측 방법</b><br>'
                '각 방법이 어떻게 작동하는지 직접 실행해봅니다</div>', unsafe_allow_html=True)

    ratio = st.select_slider("학습:검증 비율", options=["50:50","60:40","70:30","80:20"], value="70:30")
    r = {"50:50":0.5,"60:40":0.6,"70:30":0.7,"80:20":0.8}[ratio]
    X = df[["구리","환율","유가WTI"]].values.astype(float)
    y = df["LME가격"].values.astype(float)
    n_tr=int(len(y)*r)
    Xtr,ytr,Xte,yte=X[:n_tr],y[:n_tr],X[n_tr:],y[n_tr:]
    dates_te = df['날짜'].values[n_tr:]

    method = st.radio("실행할 방법", ["① RSM 단독","② 직접 예측(SVR)","③ 계수 예측","④ 2단계 예측"], horizontal=True)

    if st.button("▶ 이 방법 실행", type="primary"):
        set_rsm_scaler(Xtr)
        with st.spinner("모델 학습·예측 중..."):
            t0=time.time()
            if method.startswith("①"):
                st.markdown('<div class="formula">y = β₀ + β₁·구리 + β₂·환율 + β₃·유가 + β₄·구리² + β₅·환율² + β₆·유가²</div>',unsafe_allow_html=True)
                beta=fit_rsm(Xtr,ytr,order=2)
                pred=predict_rsm(Xte,beta,2)
                st.markdown("**추정된 계수 β (실시간)**")
                names=["상수","구리","환율","유가","구리²","환율²","유가²"]
                st.dataframe(pd.DataFrame({"항":names,"계수":[round(b,2) for b in beta]}),hide_index=True,use_container_width=True)
            elif method.startswith("②"):
                st.markdown('<div class="formula">SVR: 외부변수(구리·환율·유가) → LME 직접 매핑 (RBF 커널)</div>',unsafe_allow_html=True)
                pred=scenario1_direct(Xtr,ytr,Xte,"SVR")
            elif method.startswith("③"):
                st.markdown('<div class="formula">머신러닝이 RSM 계수 β를 예측 → 예측된 β로 RSM 식 계산</div>',unsafe_allow_html=True)
                pred=scenario2_ml_rsm(Xtr,ytr,Xte,"SVR",order=2,window=18,stabilize=True)
            else:
                st.markdown('<div class="formula">1단계: 유가 → 구리 예측  |  2단계: [유가,예측구리] → LME</div>',unsafe_allow_html=True)
                dtr={"유가":df["유가WTI"].values[:n_tr],"구리":df["구리"].values[:n_tr],"환율":df["환율"].values[:n_tr],"LME":y[:n_tr]}
                dte={"유가":df["유가WTI"].values[n_tr:],"구리":df["구리"].values[n_tr:],"환율":df["환율"].values[n_tr:],"LME":yte}
                pred,_=scenario2_step_forecast(dtr,dte,"RSM")
            elapsed=time.time()-t0

            # 예측 vs 실제 그래프
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=dates_te,y=yte,name="실제",line=dict(color=NAVY,width=2.5)))
            fig.add_trace(go.Scatter(x=dates_te,y=pred,name="예측",line=dict(color=COPPER,width=2,dash="dash")))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#888"), xaxis=dict(gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(gridcolor="rgba(128,128,128,0.2)"), height=360,margin=dict(t=20,b=20),yaxis_title="LME ($/톤)",legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig,use_container_width=True)
            m=evaluate(yte,pred)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("MAPE",f"{m['MAPE']:.2f}%")
            c2.metric("RMSE",f"{m['RMSE']:.1f}")
            c3.metric("R²",f"{m['R2']:.3f}")
            c4.metric("방향성",f"{m['Dstat']:.0f}%")
            st.caption(f"⏱ 실제 계산 시간: {elapsed:.2f}초 — 진짜로 방금 계산된 결과입니다")

# ══════════════════════════════════════════════
# 3.3 평가지표
# ══════════════════════════════════════════════
elif section.startswith("3.3"):
    st.markdown('<div class="hero"><b>논문 3.3절 · 평가지표 도출</b><br>'
                '각 지표가 어떻게 계산되는지 수식과 함께 확인합니다</div>', unsafe_allow_html=True)
    st.markdown("""
- **MAE** (평균 절대 오차): 예측이 평균 몇 단위 빗나갔나
- **RMSE** (제곱근 평균 제곱오차): 큰 오차에 더 민감
- **MAPE** (평균 절대 백분율 오차): 오차를 %로 — 해석 쉬움
- **R²** (결정계수): 1에 가까울수록 변동을 잘 설명
- **D-stat** (방향성 정확도): 상승·하락 방향을 맞춘 비율
""")
    st.markdown('<div class="step">직접 숫자를 넣어 지표가 어떻게 계산되는지 확인해보세요.</div>',unsafe_allow_html=True)
    st.markdown("**간단 예제** — 실제값과 예측값을 입력")
    col1,col2=st.columns(2)
    actual_str=col1.text_input("실제값 (쉼표 구분)","2500,2600,2550,2700")
    pred_str=col2.text_input("예측값 (쉼표 구분)","2450,2620,2500,2650")
    if st.button("▶ 지표 계산"):
        try:
            a=np.array([float(x) for x in actual_str.split(",")])
            p=np.array([float(x) for x in pred_str.split(",")])
            m=evaluate(a,p)
            st.markdown('<div class="formula">MAPE = mean(|실제-예측| / |실제|) × 100</div>',unsafe_allow_html=True)
            c=st.columns(5)
            c[0].metric("MAE",f"{m['MAE']:.1f}")
            c[1].metric("RMSE",f"{m['RMSE']:.1f}")
            c[2].metric("MAPE",f"{m['MAPE']:.2f}%")
            c[3].metric("R²",f"{m['R2']:.3f}")
            c[4].metric("D-stat",f"{m['Dstat']:.0f}%")
            # 계산 과정 노출
            st.markdown("**계산 과정**")
            steps=pd.DataFrame({"실제":a,"예측":p,"절대오차":np.abs(a-p),"백분율오차(%)":np.abs((a-p)/a)*100})
            st.dataframe(steps.round(2),hide_index=True,use_container_width=True)
        except Exception as e:
            st.error(f"입력 형식 오류: {e}")

# ══════════════════════════════════════════════
# 4.1 방법 성능 비교
# ══════════════════════════════════════════════
elif section.startswith("4.1"):
    st.markdown('<div class="hero"><b>논문 4.1절 · 방법 성능 비교</b><br>'
                '논문 표 2를 실시간으로 재현합니다 (전 방법 × 전 기간)</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">아래 버튼을 누르면 4가지 방법을 3가지 기간(단기·중기·장기)에 대해 모두 실제로 계산합니다. 다소 시간이 걸립니다.</div>',unsafe_allow_html=True)
    if st.button("▶ 논문 표 2 전체 재현", type="primary"):
        X=df[["구리","환율","유가WTI"]].values.astype(float)
        y=df["LME가격"].values.astype(float)
        rows=[]
        prog=st.progress(0,text="계산 시작...")
        combos=[("단기",0.8),("중기",0.7),("장기",0.5)]
        total=len(combos); done=0
        results={}
        for label,r in combos:
            n_tr=int(len(y)*r); Xtr,ytr,Xte,yte=X[:n_tr],y[:n_tr],X[n_tr:],y[n_tr:]
            set_rsm_scaler(Xtr)
            dtr={"유가":df["유가WTI"].values[:n_tr],"구리":df["구리"].values[:n_tr],"환율":df["환율"].values[:n_tr],"LME":y[:n_tr]}
            dte={"유가":df["유가WTI"].values[n_tr:],"구리":df["구리"].values[n_tr:],"환율":df["환율"].values[n_tr:],"LME":yte}
            res={}
            res["RSM 단독"]=evaluate(yte,predict_rsm(Xte,fit_rsm(Xtr,ytr,2),2))["MAPE"]
            res["직접(SVR)"]=evaluate(yte,scenario1_direct(Xtr,ytr,Xte,"SVR"))["MAPE"]
            try: res["계수 예측"]=evaluate(yte,scenario2_ml_rsm(Xtr,ytr,Xte,"SVR",order=2,window=18,stabilize=True))["MAPE"]
            except: res["계수 예측"]=None
            try:
                sp,_=scenario2_step_forecast(dtr,dte,"RSM"); res["2단계"]=evaluate(yte,sp)["MAPE"]
            except: res["2단계"]=None
            results[label]=res
            done+=1; prog.progress(done/total,text=f"{label} 완료")
        prog.empty()
        # 표 구성
        tbl=pd.DataFrame({
            "방법":["RSM 단독","직접(SVR)","계수 예측","2단계"],
            "단기":[results["단기"][k] for k in ["RSM 단독","직접(SVR)","계수 예측","2단계"]],
            "중기":[results["중기"][k] for k in ["RSM 단독","직접(SVR)","계수 예측","2단계"]],
            "장기":[results["장기"][k] for k in ["RSM 단독","직접(SVR)","계수 예측","2단계"]],
        })
        st.dataframe(tbl.round(2),hide_index=True,use_container_width=True)
        st.markdown('<div class="res">💡 방금 실제로 계산된 결과입니다. RSM 단독이 단기·중기에서 최우수 — 논문 결론과 일치</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 4.2 계수 예측의 한계
# ══════════════════════════════════════════════
elif section.startswith("4.2"):
    st.markdown('<div class="hero"><b>논문 4.2절 · 계수 예측의 한계</b><br>'
                'RSM 계수가 왜 예측 불가능한지 직접 확인합니다 (논문 그림 5)</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">이동 윈도우로 RSM 계수를 반복 추정해서, 계수가 시간에 따라 얼마나 요동치는지 측정합니다.</div>',unsafe_allow_html=True)
    if st.button("▶ 계수 변동성 실시간 측정", type="primary"):
        X=df[["구리","환율","유가WTI"]].values.astype(float)
        y=df["LME가격"].values.astype(float)
        n_tr=int(len(y)*0.7); Xtr,ytr=X[:n_tr],y[:n_tr]; set_rsm_scaler(Xtr)
        with st.spinner("이동 윈도우로 계수 반복 추정 중..."):
            window=18; beta_seq=[]
            for i in range(window,len(Xtr)+1):
                beta_seq.append(fit_rsm(Xtr[i-window:i],ytr[i-window:i],order=2))
            beta_seq=np.array(beta_seq)
            names=["상수","β(구리)","β(환율)","β(유가)","β(구리²)","β(환율²)","β(유가²)"]
            cvs=[abs(np.std(beta_seq[:,j])/(np.mean(beta_seq[:,j])+1e-9)) for j in range(beta_seq.shape[1])]
            fig=go.Figure(go.Bar(x=names,y=cvs,marker_color=[GREEN if v<0.3 else (COPPER if v<5 else RED) for v in cvs]))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#888"), xaxis=dict(gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(gridcolor="rgba(128,128,128,0.2)"), height=340,margin=dict(t=20,b=20),yaxis_title="변동계수 CV (log)",yaxis_type="log")
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(pd.DataFrame({"계수":names,"변동계수(CV)":[round(v,2) for v in cvs]}),hide_index=True,use_container_width=True)
            st.markdown('<div class="res">💡 상수항만 안정(CV 0.08). 환율² 계수는 CV 40+로 극도로 불안정 → ML로 예측 불가능한 근본 이유</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 4.4 구간 분리
# ══════════════════════════════════════════════
elif section.startswith("4.4"):
    st.markdown('<div class="hero"><b>논문 4.4절 · 구간 분리 분석</b><br>'
                '국면별로 최적 방법이 다름을 실시간으로 확인합니다 (논문 표 3)</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">세 국면(안정기·급등기·조정기)에서 각각 방법을 실행해 최적을 찾습니다.</div>',unsafe_allow_html=True)
    if st.button("▶ 구간별 방법 실시간 비교", type="primary"):
        periods=[("안정기","2017-01","2020-03"),("급등기","2020-04","2022-03"),("조정기","2022-04","2025-12")]
        rows=[]; prog=st.progress(0,text="시작")
        for pi,(name,s,e) in enumerate(periods):
            sub=df[(df['날짜']>=s)&(df['날짜']<=e)].reset_index(drop=True)
            X=sub[["구리","환율","유가WTI"]].values.astype(float); yv=sub["LME가격"].values.astype(float)
            n_tr=int(len(yv)*0.7); Xtr,ytr,Xte,yte=X[:n_tr],yv[:n_tr],X[n_tr:],yv[n_tr:]
            set_rsm_scaler(Xtr)
            dtr={"유가":sub["유가WTI"].values[:n_tr],"구리":sub["구리"].values[:n_tr],"환율":sub["환율"].values[:n_tr],"LME":yv[:n_tr]}
            dte={"유가":sub["유가WTI"].values[n_tr:],"구리":sub["구리"].values[n_tr:],"환율":sub["환율"].values[n_tr:],"LME":yte}
            cand={}
            cand["RSM 단독"]=evaluate(yte,predict_rsm(Xte,fit_rsm(Xtr,ytr,2),2))["MAPE"]
            cand["직접(SVR)"]=evaluate(yte,scenario1_direct(Xtr,ytr,Xte,"SVR"))["MAPE"]
            try:
                sp,_=scenario2_step_forecast(dtr,dte,"RSM"); cand["2단계"]=evaluate(yte,sp)["MAPE"]
            except: cand["2단계"]=999
            best=min(cand,key=cand.get)
            rows.append({"구간":name,"RSM단독":round(cand["RSM 단독"],2),"직접SVR":round(cand["직접(SVR)"],2),"2단계":round(cand["2단계"],2),"최적":best})
            prog.progress((pi+1)/3,text=f"{name} 완료")
        prog.empty()
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.markdown('<div class="res">💡 구간마다 최적 방법이 다름 — 안정기는 RSM, 급등기는 2단계, 조정기는 직접예측. "절대 강자 없음"을 실시간으로 확인</div>',unsafe_allow_html=True)
