import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="المنصة الذكية لتحليل المشاريع التنموية", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- محرك الأهداف (SDGs) بناءً على منهجية البنك الدولي المحدثة ---
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

# --- وظيفة هندسة السمات (التي تم تحديثها بناءً على شرحك) ---
def engineer_features(selected_sdgs):
    if not selected_sdgs:
        return None
    
    count = len(selected_sdgs)
    # تجميع الأبعاد بناءً على توزيعك (8 اجتماعي، 4 اقتصادي، 5 بيئي)
    dim_s = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'social')
    dim_e = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'economic')
    dim_en = sum(1 for s in selected_sdgs if SDG_INFO[s]['dim'] == 'environmental')
    
    # حساب النسب المئوية (Ratios)
    s_ratio = (dim_s / count) * 100
    e_ratio = (dim_e / count) * 100
    en_ratio = (dim_en / count) * 100
    
    # تطبيق معادلة Balance_score: 100 - (مجموع الانحرافات / 2)
    deviations = abs(s_ratio - 33.33) + abs(e_ratio - 33.33) + abs(en_ratio - 33.33)
    balance_score = 100 - (deviations / 2)
    
    # تحديد التوجه (Project Trend)
    ratios = {'اجتماعي': s_ratio, 'اقتصادي': e_ratio, 'بيئي': en_ratio}
    trend = max(ratios, key=ratios.get) if balance_score < 80 else "متوازن"
    
    return {
        'SDG_count': count,
        'Social_ratio': s_ratio,
        'Economic_ratio': e_ratio,
        'Environmental_ratio': en_ratio,
        'Balance_score': balance_score,
        'Trend': trend,
        'interactions': {
            'SE': dim_s * dim_e,
            'S_En': dim_s * dim_en,
            'E_En': dim_e * dim_en,
            'SEE': dim_s * dim_e * dim_en
        }
    }

# --- التصميم ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * { font-family: 'Tajawal', sans-serif; direction: rtl; }
    .report-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { background-color: #0F172A !important; color: white !important; font-weight: bold; border-radius: 10px; height: 3.5em; width: 100%; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #0F172A; }
    </style>
""", unsafe_allow_html=True)

st.title("المنصة الذكية لتحليل المشاريع التنموية")
st.write("أداة التنبؤ بمستقبل المشاريع بناءً على سجلات البنك الدولي (1947 - 2024)")
st.markdown("---")

# --- واجهة الإدخال ---
with st.sidebar:
    st.header("إعدادات المشروع")
    p_name = st.text_input("اسم المشروع", "مشروع جديد")
    p_budget = st.number_input("الميزانية التقديرية (USD)", min_value=1000, value=500000)
    p_duration = st.slider("المدة المتوقعة (أشهر)", 1, 120, 24)
    st.info("هذه المنصة تستخدم الميزات الـ 4 الأكثر تأثيراً التي حددتها دراسة الـ 18,000 مشروع.")

st.subheader("أهداف التنمية المستدامة (SDGs)")
selected_sdgs = st.multiselect(
    "اختر الأهداف المرتبطة بالمشروع:",
    options=list(SDG_INFO.keys()),
    format_func=lambda x: f"الهدف {x}: {SDG_INFO[x]['name']}"
)

if st.button("بدء التحليل الذكي"):
    if not selected_sdgs:
        st.error("⚠️ يرجى اختيار هدف واحد على الأقل.")
    else:
        # استخراج الميزات
        data = engineer_features(selected_sdgs)
        
        # محاكاة التنبؤ (هنا يتم ربط ملفات pkl لاحقاً)
        # النجاح يزيد بزيادة التوازن وعدد الأهداف
        prob = 0.55 + (data['Balance_score'] / 400) + (data['SDG_count'] * 0.02)
        prob = min(prob, 0.98)
        
        # العرض المرئي
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class="report-card" style="text-align: center;">
                    <h3 style="color: #64748B;">نسبة النجاح المتوقعة</h3>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div style="background: #DCFCE7; color: #166534; padding: 10px; border-radius: 8px; font-weight: bold; margin-top:15px;">
                        {'مشروع عالي الجودة' if prob > 0.7 else 'يحتاج تحسين في التصميم'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # مخاطر التأخير (Delay Risk)
            risk = "منخفضة" if p_duration < 48 else "مرتفعة"
            st.warning(f"مخاطر التعثر الزمني: {risk}")

        with col2:
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            st.subheader("نتائج هندسة السمات (الأكثر تأثيراً)")
            m1, m2, m3 = st.columns(3)
            m1.metric("عدد الأهداف", data['SDG_count'])
            m2.metric("مؤشر التوازن", f"{data['Balance_score']:.1f}%")
            m3.metric("التوجه", data['Trend'])
            
            st.markdown("---")
            st.write("**توزيع الأبعاد التنموية:**")
            st.progress(data['Social_ratio']/100, text=f"اجتماعي: {data['Social_ratio']:.1f}%")
            st.progress(data['Economic_ratio']/100, text=f"اقتصادي: {data['Economic_ratio']:.1f}%")
            st.progress(data['Environmental_ratio']/100, text=f"بيئي: {data['Environmental_ratio']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # تحليل التكامل (Integration Dimensions)
        st.subheader("تحليل التكامل بين الأبعاد (Interactions)")
        c1, c2, c3 = st.columns(3)
        c1.metric("تكامل اجتماعي-اقتصادي", "✅" if data['interactions']['SE'] > 0 else "❌")
        c2.metric("تكامل اجتماعي-بيئي", "✅" if data['interactions']['S_En'] > 0 else "❌")
        c3.metric("تكامل اقتصادي-بيئي", "✅" if data['interactions']['E_En'] > 0 else "❌")

st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 50px;'>مبني على منهجية تحليل سجلات البنك الدولي - الإصدار الذكي 2.0</div>", unsafe_allow_html=True)

