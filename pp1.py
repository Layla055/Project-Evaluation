import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Strategic Project Evaluator", layout="wide", initial_sidebar_state="collapsed")

# --- 2. ENGINE (HIDDEN LOGIC) ---
@st.cache_resource
def load_engine():
    try:
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except:
        return None, None, None

scaler, xgb_model, ann_model = load_engine()

def infer_internal_metrics(category):
    mapping = {
        "تعليمي": ["التعليم الجيد", "الحد من عدم المساواة"],
        "صحي": ["الصحة والرفاه", "المياه النظيفة"],
        "اجتماعي": ["القضاء على الفقر", "المجتمعات المستدامة"],
        "بيئي": ["العمل المناخي", "طاقة نظيفة"],
        "اقتصادي": ["العمل اللائق", "الصناعة والابتكار"],
        "تقني": ["الابتكار", "البنية التحتية"]
    }
    return mapping.get(category, ["عقد الشراكات"])

# --- 3. PROFESSIONAL STYLING (NAVY & GREY) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Noto+Sans+Arabic:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans Arabic', sans-serif;
        background-color: #F8FAFC;
    }
    
    .main { background-color: #F8FAFC; }
    
    /* Input Styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        color: #1E293B !important;
        padding: 12px !important;
    }
    
    /* Button Styling - Corporate Blue */
    div.stButton > button {
        background-color: #0F172A; /* Deep Navy */
        color: #F8FAFC;
        border-radius: 4px;
        width: 100%;
        height: 52px;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        transition: all 0.2s ease;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background-color: #1E293B;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }
    
    /* Titles */
    h1 { 
        color: #0F172A; 
        font-weight: 800; 
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    label {
        color: #475569 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("<div style='text-align: center; padding: 2rem 0;'><h1 dir='rtl'>محرّك تقييم الجدوى الاستراتيجية</h1></div>", unsafe_allow_html=True)

# --- 5. DATA ENTRY FORM ---
with st.container():
    with st.form("professional_entry", clear_on_submit=False):
        # Row 1
        r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
        with r1c1:
            p_name = st.text_input("اسم المشروع")
        with r1c2:
            p_cat = st.selectbox("المجال الاستراتيجي", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
        with r1c3:
            p_budget = st.number_input("الميزانية (SAR)", min_value=1000, step=5000)
            
        # Row 2
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            p_duration = st.number_input("النطاق الزمني (يوم)", min_value=1)
        with r2c2:
            p_ben = st.number_input("المستفيدون المستهدفون", min_value=1)
            
        p_desc = st.text_area("وصف المبادرة ونطاق العمل", height=100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("إصدار تقرير التحليل")

# --- 6. ANALYTICS REPORT ---
if analyze_btn:
    if not p_name or not p_desc:
        st.error("يرجى استكمال البيانات المطلوبة لتمكين التحليل.")
    elif scaler and xgb_model and ann_model:
        # Internal Calculation
        metrics = infer_internal_metrics(p_cat)
        m_count = len(metrics)
        social_factor = min(p_ben / (p_budget/100), 1.0)
        env_factor = 0.85 if p_cat == "بيئي" else 0.35
        balance = min(m_count / 4 + 0.5, 1.0)
        
        # Inference
        inputs = scaler.transform(np.array([[m_count, social_factor, balance, env_factor]]))
        res_ann = ann_model.predict(inputs).flatten()[0]
        res_xgb = xgb_model.predict_proba(inputs)[:, 1][0]
        final_index = (res_ann * 0.6) + (res_xgb * 0.4)
        
        # Derived Data
        s_roi = round(final_index * (p_ben / (p_budget/1000)), 2)
        eco_val = f"{int(p_budget * final_index * 1.35):,}"
        
        # CORPORATE UI REPORT
        report_html = f"""
        <div dir="rtl" style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 40px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top: 2rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #0F172A; padding-bottom: 15px; margin-bottom: 30px;">
                <h2 style="color: #0F172A; margin: 0;">تقرير تقييم الأثر: {p_name}</h2>
                <span style="color: #64748B; font-weight: bold;">كود التحليل: #{np.random.randint(1000,9999)}</span>
            </div>

            <div style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 40px;">
                <!-- Main Score Card -->
                <div style="background: #F8FAFC; padding: 30px; border-radius: 4px; text-align: center;">
                    <span style="color: #64748B; font-size: 0.9rem; text-transform: uppercase; font-weight: 700;">مؤشر احتمالية النجاح الاستراتيجي</span>
                    <div style="font-size: 4.5rem; font-weight: 800; color: #0F172A; margin: 10px 0;">{final_index*100:.1f}%</div>
                    <div style="height: 8px; background: #E2E8F0; border-radius: 10px; overflow: hidden; width: 80%; margin: 0 auto;">
                        <div style="width: {final_index*100}%; height: 100%; background: #0F172A;"></div>
                    </div>
                </div>

                <!-- Secondary Metrics -->
                <div style="display: grid; grid-template-rows: 1fr 1fr 1fr; gap: 15px;">
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">العائد الاجتماعي التقديري</span>
                        <span style="font-weight: 800; color: #0F172A; font-size: 1.2rem;">{s_roi}x</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">القيمة الاقتصادية المضافة</span>
                        <span style="font-weight: 800; color: #0F172A; font-size: 1.2rem;">{eco_val} ريال</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">تصنيف الاستدامة</span>
                        <span style="font-weight: 800; color: #0F172A;">{"A+" if final_index > 0.75 else "B"}</span>
                    </div>
                </div>
            </div>

            <div style="margin-top: 30px; padding: 20px; border-right: 4px solid #0F172A; background: #F1F5F9;">
                <h4 style="margin: 0 0 10px 0; color: #0F172A;">المواءمة مع المستهدفات الاستراتيجية:</h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    {' '.join([f'<span style="background: #FFFFFF; border: 1px solid #CBD5E1; padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; color: #334155;">{m}</span>' for m in metrics])}
                </div>
            </div>

            <div style="margin-top: 25px; color: #1E293B; line-height: 1.6; border: 1px solid #E2E8F0; padding: 20px;">
                <strong>الخلاصة الاستشارية:</strong><br>
                {"بناءً على النمذجة الرياضية، يتمتع المشروع بفرص واعدة للتحقق الميداني. نوصي بتخصيص الموارد المطلوبة والبدء في مرحلة التنفيذ." if final_index > 0.6 else "تشير المعطيات الحالية إلى وجود فجوة في كفاءة التشغيل مقابل الأثر. نوصي بإعادة تقييم هيكلة التكاليف."}
            </div>
        </div>
        """
        components.html(report_html, height=750, scrolling=True)
    else:
        st.error("نظام التقييم غير جاهز حالياً. يرجى التأكد من تكامل الملفات.")

# --- 7. FOOTER ---
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 3rem;'>© 2024 نظام التقييم المؤسسي المستقل</div>", unsafe_allow_html=True)

