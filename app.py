

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* Force light mode text */
    .stApp {
        background-color: #f8f9fa;
        color: #1a1a2e;
    }
    
    /* Metric labels and values */
    [data-testid="stMetricLabel"] {
        color: #1a1a2e !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    
    /* General text */
    p, h1, h2, h3, label {
        color: #1a1a2e !important;
    }
    </style>

    <style>
    /* Sidebar light theme */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }
    </style>
""", unsafe_allow_html=True)
# Page config
st.set_page_config(
    page_title="PCP Disengagement Risk Scorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/outputs/member_feature.csv')
    sp_cols = ['SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR',
               'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES',
               'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']
    if df[sp_cols].max().max() == 2:
        df[sp_cols] = df[sp_cols].replace({1: 1, 2: 0})
    return df

df = load_data()

sp_cols = ['SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR',
           'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES',
           'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']

condition_names = {
    'SP_ALZHDMTA': 'Alzheimers',
    'SP_CHF': 'Heart Failure',
    'SP_CHRNKIDN': 'Kidney Disease',
    'SP_CNCR': 'Cancer',
    'SP_COPD': 'COPD',
    'SP_DEPRESSN': 'Depression',
    'SP_DIABETES': 'Diabetes',
    'SP_ISCHMCHT': 'Ischemic Heart',
    'SP_OSTEOPRS': 'Osteoporosis',
    'SP_RA_OA': 'Arthritis',
    'SP_STRKETIA': 'Stroke/TIA'
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("MediCore Analytics")
    st.markdown("**AEP Outreach Dashboard**")
    st.markdown("---")
    
    selected_tier = st.selectbox(
        "Filter by Risk Tier",
        ["All", "High", "Medium", "Low"]
    )
    
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("Accuracy", "71.81%")
    st.metric("AUC-ROC", "0.78")
    st.metric("Total Members", f"{len(df):,}")

# Filter data
if selected_tier != "All":
    filtered_df = df[df['RISK_TIERS'] == selected_tier]
else:
    filtered_df = df

# Header
st.title("🏥 PCP Disengagement Risk Scorer")
st.markdown("**Pre-AEP Member Risk Intelligence Dashboard - MediCore Advantage Health Plan**")
st.markdown("---")

# KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Members",
        f"{len(df):,}",
        delta=None
    )

with col2:
    high_count = len(df[df['RISK_TIERS'] == 'High'])
    st.metric(
        "🔴 High Risk",
        f"{high_count:,}",
        delta="Immediate Outreach"
    )

with col3:
    med_count = len(df[df['RISK_TIERS'] == 'Medium'])
    st.metric(
        "🟡 Medium Risk",
        f"{med_count:,}",
        delta="Monitor Closely"
    )

with col4:
    disengagement_rate = df['DISENGAGED'].mean() * 100
    st.metric(
        "Disengagement Rate",
        f"{disengagement_rate:.1f}%",
        delta=None
    )

st.markdown("---")

# Row 1 - Risk Distribution
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Tier Breakdown")
    tier_counts = df['RISK_TIERS'].value_counts()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = {
        'High': '#ef4444',
        'Medium': '#f59e0b',
        'Low': '#22c55e'
    }
    wedge_colors = [colors[t] for t in tier_counts.index]
    wedges, texts, autotexts = ax.pie(
        tier_counts,
        labels=tier_counts.index,
        autopct='%1.1f%%',
        colors=wedge_colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title("Member Distribution", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Geographic Risk — Top 10 States")
    state_risk = df[df['RISK_TIERS'] == 'High'].groupby(
        'SP_STATE_CODE').size().reset_index(name='high_risk_count')
    state_risk = state_risk.sort_values(
        'high_risk_count', ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        state_risk['SP_STATE_CODE'].astype(str),
        state_risk['high_risk_count'],
        color='#ef4444',
        edgecolor='white',
        height=0.6
    )
    ax.set_xlabel("High Risk Member Count", fontsize=9)
    ax.set_title("High Risk Members by State Code", 
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}',
                va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Row 2 - Chronic Conditions + Risk Score Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Chronic Conditions — High Risk Members")
    high_risk_df = df[df['RISK_TIERS'] == 'High']
    condition_rates = high_risk_df[sp_cols].mean()
    condition_rates.index = [condition_names[c] for c in condition_rates.index]
    condition_rates = condition_rates.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(condition_rates.index, condition_rates.values,
                   color='#f59e0b', edgecolor='white', height=0.6)
    ax.set_xlabel("Prevalence Rate", fontsize=9)
    ax.set_title("Condition Prevalence", fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Risk Score Distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    colors_map = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}
    for tier in ['Low', 'Medium', 'High']:
        tier_data = df[df['RISK_TIERS'] == tier]['RISK_SCORE']
        ax.hist(tier_data, bins=30, alpha=0.7,
                label=tier, color=colors_map[tier], edgecolor='white')
    
    ax.set_xlabel("Risk Score", fontsize=9)
    ax.set_ylabel("Number of Members", fontsize=9)
    ax.set_title("Risk Score Distribution by Tier",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Row 3 - High Risk Member Table
st.subheader("🔴 Priority Outreach List - High Risk Members")

high_risk_table = df[df['RISK_TIERS'] == 'High'][
    ['DESYNPUF_ID', 'RISK_SCORE', 'chronic_burden',
     'age', 'SP_STATE_CODE', 'MEDREIMB_CAR', 'MEDREIMB_OP']
].sort_values('RISK_SCORE', ascending=False).head(100)

high_risk_table.columns = [
    'Member ID', 'Risk Score', 'Chronic Conditions',
    'Age', 'State Code', 'CAR Spend', 'OP Spend'
]

st.dataframe(
    high_risk_table,
    use_container_width=True,
    height=300
)

csv = high_risk_table.to_csv(index=False)
st.download_button(
    label="⬇️ Download High Risk Member List",
    data=csv,
    file_name="high_risk_members_AEP.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("MediCore Analytics | PCP Disengagement Risk Model | AUC-ROC: 0.78 | Built with CMS SynPUF Data")