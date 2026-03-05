import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. إعدادات الهوية البصرية الرسمية ---
st.set_page_config(page_title="Strategic Project Evaluator", layout="wide", initial_sidebar_state="collapsed")

# --- 2. محرك التحليل (مخفي تماماً عن المستخدم) ---
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

# --- 3. تصميم الواجهة (احترافي: نيلي ورمادي وفراغات رصينة) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans Arabic', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* Input Styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        color: #1E293B !important;
        padding: 12px !important;
    }
    
    /* Button Styling - Navy Blue */
    div.stButton > button {
        background-color: #0F172A; /* Navy Midnight */
        color: #F8FAFC;
        border-radius: 4px;
        width: 100%;
        height: 52px;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1E293B;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.1);
    }
    
    h1 { color: #0F172A; font-weight: 800; text-align: center; margin-bottom: 2rem; }
    label { color: #475569 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. العنوان الرئيسي ---
st.markdown("<h1 dir='rtl'>نظام تقييم الجدوى والمواءمة الاستراتيجية</h1>", unsafe_allow_html=True)

# --- 5. مدخلات المشروع (منظمة وواضحة) ---
with st.container():
    with st.form("professional_form", clear_on_submit=False):
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            p_name = st.text_input("اسم المشروع")
            p_desc = st.text_area("وصف المبادرة", height=150, placeholder="اكتب نبذة مختصرة عن المشروع...")
            
        with col_side:
            p_cat = st.selectbox("المجال الاستراتيجي", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
            p_budget = st.number_input("الميزانية التقديرية (SAR)", min_value=1000, step=1000)
            p_duration = st.number_input("النطاق الزمني (يوم)", min_value=1)
            p_ben = st.number_input("المستفيدون المستهدفون", min_value=1)
            
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.form_submit_button("إصدار تقرير التحليل النهائي")

# --- 6. عرض النتائج (Dashboard Style) ---
if analyze_btn:
    if not p_name or not p_desc:
        st.error("يرجى إكمال البيانات الأساسية لتفعيل محرك التقييم.")
    elif scaler and xgb_model and ann_model:
        # العمليات الداخلية (مخفية)
        metrics = infer_internal_metrics(p_cat)
        m_count = len(metrics)
        social_f = min(p_ben / (p_budget/100), 1.0)
        env_f = 0.85 if p_cat == "بيئي" else 0.35
        balance = min(m_count / 4 + 0.5, 1.0)
        
        # التنبؤ
        inputs = scaler.transform(np.array([[m_count, social_f, balance, env_f]]))
        res_ann = ann_model.predict(inputs).flatten()[0]
        res_xgb = xgb_model.predict_proba(inputs)[:, 1][0]
        score = (res_ann * 0.6) + (res_xgb * 0.4)
        
        # النتائج المشتقة
        s_roi = round(score * (p_ben / (p_budget/1000)), 2)
        eco_val = f"{int(p_budget * score * 1.3):,}"
        
        # تصميم التقرير الرسمي
        report_ui = f"""
        <div dir="rtl" style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 40px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top: 2rem;">
            <div style="border-bottom: 2px solid #0F172A; padding-bottom: 15px; margin-bottom: 30px;">
                <h2 style="color: #0F172A; margin: 0;">تقرير التقييم الاستراتيجي: {p_name}</h2>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: start;">
                <!-- مؤشر النجاح -->
                <div style="background: #F8FAFC; padding: 30px; border-radius: 4px; text-align: center; border: 1px solid #E2E8F0;">
                    <span style="color: #64748B; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">احتمالية النجاح الاستراتيجية</span>
                    <div style="font-size: 4rem; font-weight: 800; color: #0F172A; margin: 15px 0;">{score*100:.1f}%</div>
                    <div style="height: 8px; background: #E2E8F0; border-radius: 10px; overflow: hidden; width: 80%; margin: 0 auto;">
                        <div style="width: {score*100}%; height: 100%; background: #0F172A;"></div>
                    </div>
                </div>

                <!-- مصفوفة الأثر -->
                <div style="display: grid; gap: 15px;">
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between;">
                        <span style="color: #475569;">العائد الاجتماعي المتوقع</span>
                        <span style="font-weight: 800; color: #0F172A;">{s_roi}x</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between;">
                        <span style="color: #475569;">القيمة الاقتصادية المضافة</span>
                        <span style="font-weight: 800; color: #0F172A;">{eco_val} ريال</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px; display: flex; justify-content: space-between;">
                        <span style="color: #475569;">تصنيف الاستدامة</span>
                        <span style="font-weight: 800; color: #0F172A;">{"متفوق (A)" if score > 0.7 else "مقبول (B)"}</span>
                    </div>
                </div>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: #F1F5F9; border-right: 5px solid #0F172A;">
                <h4 style="margin: 0 0 10px 0; color: #0F172A;">المواءمة مع المستهدفات:</h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    {' '.join([f'<span style="background: white; border: 1px solid #CBD5E1; padding: 4px 12px; border-radius: 4px; font-size: 0.85rem;">{m}</span>' for m in metrics])}
                </div>
            </div>

            <div style="margin-top: 25px; padding: 20px; border: 1px solid #E2E8F0; line-height: 1.6;">
                <strong>الخلاصة التقديرية:</strong><br>
                {"بناءً على التحليل المتقدم للمؤشرات، يمتلك المشروع مقومات نجاح عالية وقدرة على تحقيق أثر مستدام." if score > 0.6 else "يوصى بإعادة مراجعة الموارد التشغيلية لضمان كفاءة أعلى في تحقيق المخرجات."}
            </div>
        </div>
        """
        components.html(report_ui, height=750, scrolling=True)
    else:
        st.error("المحرك الذكي غير متصل حالياً.")

st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 4rem;'>© جميع الحقوق محفوظة - نظام تقييم الأثر</div>", unsafe_allow_html=True)

