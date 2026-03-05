import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="محلل المشاريع", layout="wide", initial_sidebar_state="collapsed")

# --- 2. محرك التحليل (مخفي عن المستخدم) ---
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

def get_internal_sdgs(cat):
    # ربط داخلي صامت للأهداف
    mapping = {
        "تعليمي": ["4. التعليم الجيد", "17. الشراكات"],
        "صحي": ["3. الصحة والرفاه"],
        "اجتماعي": ["1. القضاء على الفقر", "10. الحد من عدم المساواة"],
        "بيئي": ["13. العمل المناخي", "7. طاقة نظيفة"],
        "اقتصادي": ["8. العمل اللائق", "9. الابتكار"],
        "تقني": ["9. الصناعة والابتكار"]
    }
    return mapping.get(cat, ["17. عقد الشراكات"])

# --- 3. تصميم الواجهة (أزرق غامق ورمادي) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    div.stButton > button {
        background-color: #0d47a1; color: white; border-radius: 12px;
        width: 100%; height: 55px; font-size: 1.1em; font-weight: bold;
        border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 10px !important; border: 1px solid #cfd8dc !important;
    }
    h1 { color: #0d47a1; text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

st.title("منصة تقييم المشاريع 🛡️")

# --- 4. نموذج المدخلات ---
with st.container():
    with st.form("project_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("📝 اسم المشروع", placeholder="أدخل اسم المبادرة")
            p_cat = st.selectbox("📂 مجال المشروع", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
            p_budget = st.number_input("💰 الميزانية التقديرية (ريال)", min_value=1000, value=10000)
        with col2:
            p_duration = st.number_input("⏳ مدة المشروع (أيام)", min_value=1, value=365)
            p_ben = st.number_input("👥 عدد المستفيدين المتوقع", min_value=1, value=100)
            
        p_desc = st.text_area("💡 فكرة المشروع", placeholder="اكتب وصفاً موجزاً للمشروع وأهدافه الرئيسية")
        
        analyze_btn = st.form_submit_button("تحليل المشروع 🚀")

# --- 5. عرض النتائج الفني ---
if analyze_btn:
    if not p_name or not p_desc:
        st.error("الرجاء إكمال كافة البيانات المطلوبة.")
    elif scaler and xgb_model and ann_model:
        # حسابات داخلية
        sdgs = get_internal_sdgs(p_cat)
        sdg_count = len(sdgs)
        social_ratio = min(p_ben / (p_budget/100), 1.0)
        balance_score = min(sdg_count / 5 + 0.4, 1.0)
        env_val = 0.8 if p_cat == "بيئي" else 0.4
        
        # التنبؤ (بدون عرض اسم الموديل)
        features = scaler.transform(np.array([[sdg_count, social_ratio, balance_score, env_val]]))
        score = (ann_model.predict(features).flatten()[0] * 0.6) + (xgb_model.predict_proba(features)[:, 1][0] * 0.4)
        
        # حساب العوائد
        sroi = round(score * (p_ben / (p_budget/1000)), 2)
        eco_impact = f"{int(p_budget * score * 1.4):,}"
        
        color = "#1b5e20" if score > 0.6 else "#b71c1c"
        
        # واجهة النتائج
        ui_html = f"""
        <div dir="rtl" style="background: white; padding: 30px; border-radius: 20px; border: 1px solid #e0e0e0; box-shadow: 0 10px 25px rgba(0,0,0,0.05);">
            <h2 style="color: #0d47a1; text-align: center; margin-bottom: 25px;">تقرير التقييم: {p_name}</h2>
            
            <div style="text-align: center; margin-bottom: 40px;">
                <div style="display: inline-block; padding: 25px; border-radius: 50%; background: #f1f8ff; border: 10px solid #0d47a1; width: 180px; height: 180px;">
                    <span style="font-size: 0.85em; color: #546e7a; display: block;">احتمالية النجاح</span>
                    <span style="font-size: 2.8em; font-weight: bold; color: {color};">{score*100:.1f}%</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #eceff1;">
                    <span style="color: #78909c; font-size: 0.9em;">العائد الاجتماعي</span><br>
                    <span style="font-size: 1.6em; font-weight: bold; color: #0d47a1;">{sroi}x</span>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #eceff1;">
                    <span style="color: #78909c; font-size: 0.9em;">الأثر الاقتصادي</span><br>
                    <span style="font-size: 1.4em; font-weight: bold; color: #0d47a1;">{eco_impact} ريال</span>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #eceff1;">
                    <span style="color: #78909c; font-size: 0.9em;">الأثر البيئي</span><br>
                    <span style="font-size: 1.4em; font-weight: bold; color: #0d47a1;">{"مرتفع" if env_val > 0.5 else "متوازن"}</span>
                </div>
            </div>

            <div style="background: #ffffff; padding: 18px; border-radius: 12px; border-right: 8px solid #4caf50; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">🌍 أهداف التنمية المرتبطة:</h4>
                <p style="margin: 0; font-weight: bold; color: #424242;">{' | '.join(sdgs)}</p>
            </div>

            <div style="background: #f1f5f9; padding: 25px; border-radius: 15px; border-right: 8px solid #0d47a1;">
                <h4 style="margin: 0 0 10px 0; color: #0d47a1;">💡 توصية التقييم:</h4>
                <p style="margin: 0; line-height: 1.6; color: #37474f;">
                    {"بناءً على المعايير المدخلة، يظهر المشروع كفاءة عالية في تحقيق الأثر المطلوب ونوصي بدعمه." if score > 0.6 else "نوصي بمراجعة هيكلة التكاليف مقابل عدد المستفيدين لضمان استدامة أكبر للمشروع."}
                </p>
            </div>
        </div>
        """
        components.html(ui_html, height=850, scrolling=True)
        if score > 0.6: st.balloons()
    else:
        st.info("جاري تهيئة محرك التحليل... يرجى التأكد من رفع ملفات النماذج.")

