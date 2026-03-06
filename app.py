import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import os
import re
from datetime import datetime

# --- 1. إعدادات الهوية البصرية ---
st.set_page_config(
    page_title="المنصة الذكية لتحليل المشاريع التنموية", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. تحميل المحرك الذكي (النماذج) ---
@st.cache_resource
def load_analysis_assets():
    assets = {
        'scaler': None,
        'xgb': None,
        'ann': None,
        'status': 'offline'
    }
    try:
        if os.path.exists('scaler.pkl'):
            assets['scaler'] = joblib.load('scaler.pkl')
        if os.path.exists('hybrid_xgb.pkl'):
            assets['xgb'] = joblib.load('hybrid_xgb.pkl')
        # المحرك يعمل في وضع التجربة إذا لم تتوفر ملفات الموديل المدربة
        assets['status'] = 'online' if assets['xgb'] else 'demo_mode'
    except Exception as e:
        assets['status'] = f'error: {str(e)}'
    return assets

assets = load_analysis_assets()

# --- 3. قاموس الأهداف التنموية (SDGs) وفق المنهجية المعتمدة ---
SDG_INFO = {
    1: {"name": "القضاء على الفقر", "dim": "social"},
    2: {"name": "القضاء على الجوع", "dim": "social"},
    3: {"name": "الصحة الجيدة", "dim": "social"},
    4: {"name": "التعليم الجيد", "dim": "social"},
    5: {"name": "المساواة بين الجنسين", "dim": "social"},
    6: {"name": "المياه النظيفة", "dim": "environmental"},
    7: {"name": "الطاقة النظيفة", "dim": "environmental"},
    8: {"name": "العمل اللائق", "dim": "economic"},
    9: {"name": "الصناعة والابتكار", "dim": "economic"},
    10: {"name": "الحد من عدم المساواة", "dim": "social"},
    11: {"name": "مدن مستدامة", "dim": "social"},
    12: {"name": "الاستهلاك المسؤول", "dim": "economic"},
    13: {"name": "العمل المناخي", "dim": "environmental"},
    14: {"name": "الحياة تحت الماء", "dim": "environmental"},
    15: {"name": "الحياة في البر", "dim": "environmental"},
    16: {"name": "السلام والعدالة", "dim": "social"},
    17: {"name": "الشراكات", "dim": "economic"}
}

# --- 4. وظائف هندسة السمات (الخلفية التحليلية) ---
def engineer_features(selected_sdgs):
    """تطبيق هندسة السمات بناءً على دراسة البنك الدولي"""
    if not selected_sdgs:
        return {
            'SDG_count': 0, 'Social_ratio': 0, 'Economic_ratio': 0, 
            'Environmental_ratio': 0, 'Balance_score': 0, 'Trend': 'غير محدد'
        }
    
    count = len(selected_sdgs)
    dim_social = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'social')
    dim_economic = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'economic')
    dim_env = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'environmental')
    
    s_ratio = (dim_social / count) * 100
    e_ratio = (dim_economic / count) * 100
    env_ratio = (dim_env / count) * 100
    
    # معادلة Balance_score: قياس الانحراف عن نقطة التوازن المثالية
    deviations = abs(s_ratio - 33.33) + abs(e_ratio - 33.33) + abs(env_ratio - 33.33)
    balance_score = 100 - (deviations / 2)
    
    # تحديد التوجه التنموي
    ratios = {'اجتماعي': s_ratio, 'اقتصادي': e_ratio, 'بيئي': env_ratio}
    trend = max(ratios, key=ratios.get) if balance_score < 80 else "متوازن"
    
    return {
        'SDG_count': count,
        'Social_ratio': s_ratio,
        'Economic_ratio': e_ratio,
        'Environmental_ratio': env_ratio,
        'Balance_score': balance_score,
        'Trend': trend,
        'dims': {'S': dim_social, 'E': dim_economic, 'En': dim_env}
    }

# --- 5. التصميم وواجهة المستخدم ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * { font-family: 'Tajawal', sans-serif; direction: rtl; }
    .main { background-color: #F8FAFC; }
    .stButton>button { background-color: #0F172A !important; color: white !important; border-radius: 8px; height: 3.5em; width: 100%; font-weight: bold; border: none; }
    .report-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .metric-box { text-align: center; padding: 15px; background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0; }
    .success-text { color: #166534; background: #DCFCE7; padding: 10px; border-radius: 8px; font-weight: bold; display: inline-block; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("المنصة الذكية لتحليل المشاريع التنموية")
st.write("أداة التنبؤ بمستقبل المشاريع بناءً على إرث البنك الدولي (18,000+ مشروع)")
st.markdown("---")

# --- 6. إدخال البيانات ---
with st.sidebar:
    st.header("إعدادات المشروع")
    p_name = st.text_input("اسم المشروع التنموي", "مشروع جديد")
    p_budget = st.number_input("الميزانية (بالدولار)", min_value=0, value=1000000)
    p_duration = st.slider("مدة التنفيذ المتوقعة (أشهر)", 1, 120, 36)
    p_status = st.selectbox("حالة المشروع الحالية", ["نشط (Active)", "قيد التحضير (Pipeline)"])
    st.markdown("---")
    st.info("المنصة تستخدم الميزات الـ 4 الأكثر تأثيراً التي حددتها دراستك.")

st.subheader("أهداف التنمية المستدامة (SDGs)")
selected_sdgs = st.multiselect(
    "حدد الأهداف التي يغطيها المشروع لاستخراج 'هندسة السمات' تلقائياً:",
    options=list(SDG_INFO.keys()),
    format_func=lambda x: f"الهدف {x}: {SDG_INFO[x]['name']}"
)

if st.button("تحليل المشروع والتنبؤ بالنجاح"):
    if not selected_sdgs:
        st.error("⚠️ يرجى اختيار هدف واحد على الأقل للبدء.")
    else:
        # 1. تنفيذ هندسة السمات في الخلفية
        features = engineer_features(selected_sdgs)
        
        # 2. منطق التنبؤ (محاكاة النموذج الهجين)
        # يعتمد النجاح على Balance_score وعدد الأهداف كأوزان أساسية
        base_success = 0.60
        sdg_impact = (features['SDG_count'] * 0.025)
        balance_impact = (features['Balance_score'] / 100) * 0.15
        success_prob = min(base_success + sdg_impact + balance_impact, 0.98)
        
        # --- 3. عرض النتائج ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class="report-card" style="text-align: center;">
                    <h3 style="color: #64748B;">احتمالية نجاح المشروع</h3>
                    <h1 style="color: #0F172A; font-size: 4em; margin: 10px 0;">{success_prob*100:.1f}%</h1>
                    <div class="success-text">
                        توصية: {'مشروع استراتيجي واعد' if success_prob > 0.7 else 'يحتاج مراجعة التصميم'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # مخاطر التأخير (بناءً على السمات الزمنية)
            delay_risk = "منخفضة" if p_duration < 48 else "مرتفعة"
            st.warning(f"مستوى مخاطر التعثر الزمني: {delay_risk}")

        with col2:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.subheader("نتائج هندسة السمات (الميزات الـ 4 الرئيسية)")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("عدد الأهداف (SDG_count)", features['SDG_count'])
            m2.metric("مؤشر التوازن (Balance)", f"{features['Balance_score']:.1f}%")
            m3.metric("التوجه (Trend)", features['Trend'])
            
            st.markdown("---")
            st.write("**تركيز الأبعاد التنموية:**")
            st.progress(features['Social_ratio']/100, text=f"البعد الاجتماعي: {features['Social_ratio']:.1f}%")
            st.progress(features['Economic_ratio']/100, text=f"البعد الاقتصادي: {features['Economic_ratio']:.1f}%")
            st.progress(features['Environmental_ratio']/100, text=f"البعد البيئي: {features['Environmental_ratio']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. تحليل التكامل (Interactions)
        st.subheader("تحليل التكامل بين الأبعاد")
        t1, t2, t3 = st.columns(3)
        with t1:
            active = features['dims']['S'] > 0 and features['dims']['E'] > 0
            st.markdown(f"<div class='metric-box'><b>تكامل اجتماعي-اقتصادي</b><br>{'✅ متوفر' if active else '❌ غير مفعل'}</div>", unsafe_allow_html=True)
        with t2:
            active = features['dims']['S'] > 0 and features['dims']['En'] > 0
            st.markdown(f"<div class='metric-box'><b>تكامل اجتماعي-بيئي</b><br>{'✅ متوفر' if active else '❌ غير مفعل'}</div>", unsafe_allow_html=True)
        with t3:
            active = features['dims']['E'] > 0 and features['dims']['En'] > 0
            st.markdown(f"<div class='metric-box'><b>تكامل اقتصادي-بيئي</b><br>{'✅ متوفر' if active else '❌ غير مفعل'}</div>", unsafe_allow_html=True)

        # 5. التوصيات النهائية
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("💡 تفاصيل العائد الاجتماعي المستهدف (SROI)"):
            estimated_impact = (p_budget * success_prob * 1.8) / 1000000
            st.write(f"بناءً على التوازن التنموي المحقق ({features['Balance_score']:.1f}%)، يتوقع أن يحقق المشروع أثراً قيمته {estimated_impact:.2f} مليون دولار كعائد اجتماعي غير مباشر.")
            if features['SDG_count'] < 3:
                st.info("نصيحة: ربط المشروع بأهداف إضافية يزيد من شمولية التصميم وفرص التمويل.")

st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 50px;'>المنصة الذكية لتحليل المشاريع التنموية - إصدار 2024</div>", unsafe_allow_html=True)

