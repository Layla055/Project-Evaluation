import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. إعدادات الهوية المؤسسية ---
st.set_page_config(page_title="نظام تقييم الجدوى الاستراتيجية", layout="wide", initial_sidebar_state="collapsed")

# --- 2. محرك التحليل الذكي (خلفية برمجية صامتة) ---
@st.cache_resource
def load_analysis_engine():
    try:
        # تحميل النماذج الذكية من الملفات المرفقة
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except:
        return None, None, None

scaler, xgb_model, ann_model = load_analysis_engine()

def get_strategic_alignment(category):
    # ربط داخلي للأهداف بناءً على المجال المحدد
    alignment_map = {
        "تعليمي": ["تطوير الكفاءات البشرية", "تعزيز فرص التعلم"],
        "صحي": ["تحسين جودة الحياة", "تعزيز الرفاه الصحي"],
        "اجتماعي": ["تمكين المجتمعات", "الحد من أوجه التفاوت"],
        "بيئي": ["الاستدامة البيئية", "كفاءة الموارد"],
        "اقتصادي": ["النمو الاقتصادي المستدام", "الابتكار المؤسسي"],
        "تقني": ["التحول الرقمي", "البنية التحتية الذكية"]
    }
    return alignment_map.get(category, ["التنمية المستدامة"])

# --- 3. لغة التصميم (نيلي عميق ورمادي احترافي) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans Arabic', sans-serif;
        background-color: #F1F5F9;
    }
    
    /* تنسيق الحقول */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 2px !important;
        color: #1E293B !important;
        padding: 10px !important;
    }
    
    /* زر التحليل الرسمي */
    div.stButton > button {
        background-color: #0F172A !important; /* نيلي غامق */
        color: #F8FAFC !important;
        border-radius: 2px !important;
        width: 100% !important;
        height: 50px !important;
        font-weight: 700 !important;
        border: none !important;
        font-size: 1.1rem !important;
    }
    div.stButton > button:hover {
        background-color: #1E293B !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    h1 { color: #0F172A; font-weight: 800; text-align: center; margin-bottom: 30px; }
    label { color: #334155 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. واجهة إدخال البيانات ---
st.markdown("<h1 dir='rtl'>منصة تقييم الجدوى ومواءمة المشاريع</h1>", unsafe_allow_html=True)

with st.container():
    with st.form("professional_assessment_form"):
        # توزيع المدخلات بشكل متوازن
        c1, c2 = st.columns([1, 1])
        with c1:
            p_name = st.text_input("اسم المشروع")
            p_cat = st.selectbox("مجال المشروع", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
            p_budget = st.number_input("الميزانية التقديرية (SAR)", min_value=1000, step=1000)
        with c2:
            p_duration = st.number_input("المدة الزمنية (يوم)", min_value=1, value=365)
            p_ben = st.number_input("عدد المستفيدين المتوقع", min_value=1, value=100)
            st.write("") # فاصل
            
        p_desc = st.text_area("فكرة المشروع")
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_submit = st.form_submit_button("إجراء التحليل الاستراتيجي")

# --- 5. تقرير النتائج الاحترافي (Dashboard Style) ---
if btn_submit:
    if not p_name or not p_desc:
        st.error("يرجى تزويد النظام بالبيانات الأساسية للمشروع.")
    elif scaler and xgb_model and ann_model:
        # حسابات الذكاء الاصطناعي (في الخلفية)
        internal_targets = get_strategic_alignment(p_cat)
        target_count = len(internal_targets)
        social_weight = min(p_ben / (p_budget/100), 1.0)
        env_weight = 0.9 if p_cat == "بيئي" else 0.4
        balance_index = min(target_count / 4 + 0.5, 1.0)
        
        # تنفيذ التنبؤ الهجين
        scaled_data = scaler.transform(np.array([[target_count, social_weight, balance_index, env_weight]]))
        ann_pred = ann_model.predict(scaled_data).flatten()[0]
        xgb_pred = xgb_model.predict_proba(scaled_data)[:, 1][0]
        final_success_rate = (ann_pred * 0.6) + (xgb_pred * 0.4)
        
        # حساب العوائد التنموية
        social_roi = round(final_success_rate * (p_ben / (p_budget/1000)), 2)
        financial_impact = f"{int(p_budget * final_success_rate * 1.3):,}"
        
        # تصميم مخرجات التقرير
        report_template = f"""
        <div dir="rtl" style="background: white; border: 1px solid #E2E8F0; padding: 40px; border-radius: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px;">
            <div style="border-bottom: 3px solid #0F172A; padding-bottom: 10px; margin-bottom: 30px;">
                <h2 style="color: #0F172A; margin: 0;">ملخص تقييم المشروع: {p_name}</h2>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px;">
                <!-- مؤشر النجاح الاستراتيجي -->
                <div style="background: #F8FAFC; padding: 25px; border-radius: 2px; text-align: center; border: 1px solid #CBD5E1;">
                    <span style="color: #64748B; font-weight: 700; font-size: 0.9rem;">تنبؤ احتمالية النجاح</span>
                    <div style="font-size: 4.5rem; font-weight: 800; color: #0F172A; margin: 10px 0;">{final_success_rate*100:.1f}%</div>
                    <div style="height: 6px; background: #E2E8F0; width: 80%; margin: 0 auto;">
                        <div style="width: {final_success_rate*100}%; height: 100%; background: #0F172A;"></div>
                    </div>
                </div>

                <!-- مصفوفة الأثر المترتب -->
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="border: 1px solid #E2E8F0; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">العائد الاجتماعي</span>
                        <span style="font-weight: 800; color: #0F172A; font-size: 1.2rem;">{social_roi}x</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">العائد الاقتصادي التقديري</span>
                        <span style="font-weight: 800; color: #0F172A; font-size: 1.1rem;">{financial_impact} ريال</span>
                    </div>
                    <div style="border: 1px solid #E2E8F0; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #475569;">الأثر البيئي المتوقع</span>
                        <span style="font-weight: 800; color: #0F172A;">{"مستدام" if env_weight > 0.5 else "متوازن"}</span>
                    </div>
                </div>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: #F1F5F9; border-right: 6px solid #0F172A;">
                <h4 style="margin: 0 0 10px 0; color: #0F172A;">مواءمة الأهداف الاستراتيجية:</h4>
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    {' '.join([f'<span style="background: white; border: 1px solid #CBD5E1; padding: 4px 15px; font-size: 0.85rem; color: #334155;">{m}</span>' for m in internal_targets])}
                </div>
            </div>

            <div style="margin-top: 30px; border: 1px solid #E2E8F0; padding: 20px; line-height: 1.7;">
                <strong>التوصيات الختامية:</strong><br>
                <p style="color: #334155; margin-top: 10px;">
                {"بناءً على معطيات التحليل، يظهر المشروع كفاءة تشغيلية ومواءمة استراتيجية قوية، مما يجعله مرشحاً للدعم والتنفيذ." if final_success_rate > 0.6 else "المعطيات الحالية تشير إلى ضرورة تحسين كفاءة الوصول للأثر مقابل التكلفة المرصودة لضمان استدامة المشروع."}
                </p>
            </div>
        </div>
        """
        components.html(report_template, height=800, scrolling=True)
    else:
        st.warning("تعذر تشغيل محرك التحليل. يرجى التحقق من وجود ملفات النماذج في المستودع.")

# --- 6. التذييل ---
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 50px;'>نظام التقييم المؤسسي الموحد © 2024</div>", unsafe_allow_html=True)

