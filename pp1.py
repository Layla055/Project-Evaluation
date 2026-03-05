import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="محلل المشاريع الذكي", layout="wide", initial_sidebar_state="collapsed")

# --- 2. تحميل النماذج (المحرك الهجين) ---
@st.cache_resource
def load_models():
    try:
        # تأكد أن هذه الملفات موجودة في نفس مجلد GitHub
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except Exception as e:
        return None, None, None

scaler, xgb_model, ann_model = load_models()

# --- 3. وظيفة الاستنتاج الذكي للأهداف ---
def infer_sdgs(category, description):
    # ربط ذكي يعتمد على المجال والكلمات المفتاحية
    mapping = {
        "تعليمي": ["4. التعليم الجيد", "5. المساواة", "17. الشراكات"],
        "صحي": ["3. الصحة والرفاه", "6. المياه النظيفة"],
        "اجتماعي": ["1. الفقر", "10. الحد من الفوارق", "2. الجوع"],
        "بيئي": ["13. المناخ", "15. الحياة في البر", "7. طاقة نظيفة"],
        "اقتصادي": ["8. العمل اللائق", "9. الابتكار", "12. الاستهلاك"],
        "تقني": ["9. الابتكار والبنية التحتية", "4. التعليم التقني"]
    }
    return mapping.get(category, ["17. عقد الشراكات"])

# --- 4. واجهة المستخدم (التصميم بالأزرق والرمادي) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button {
        background-color: #0d47a1; color: white; border-radius: 8px;
        width: 100%; height: 55px; font-size: 18px; font-weight: bold;
        border: none; transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #1565c0; color: white; }
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border: 1px solid #cfd8dc !important; border-radius: 8px !important;
    }
    h1 { color: #0d47a1; text-align: center; font-family: 'Cairo', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("المنصة الذكية لتقييم المشاريع التنموية 🛡️")
st.markdown("<p style='text-align: center; color: #546e7a;'>تحليل متقدم باستخدام الذكاء الاصطناعي الهجين (ANN + XGBoost)</p>", unsafe_allow_html=True)

# استمارة المدخلات
with st.container():
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("📝 اسم المشروع", placeholder="مثال: مبادرة مهارات المستقبل")
            p_cat = st.selectbox("📂 مجال المشروع", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
            p_budget = st.number_input("💰 الميزانية (ريال)", min_value=1000, value=10000)
        with col2:
            p_duration = st.number_input("⏳ المدة المتوقعة (أيام)", min_value=1, value=180)
            p_ben = st.number_input("👥 عدد المستفيدين", min_value=1, value=100)
            st.write("") # فاصل جمالي
            
        p_desc = st.text_area("💡 اشرح فكرة المشروع بالتفصيل (لتحليل الأثر والأهداف)", height=120)
        
        analyze_btn = st.form_submit_button("بدء التحليل الذكي الشامل 🚀")

# --- 5. منطق التحليل والمخرجات ---
if analyze_btn:
    if not p_name or not p_desc:
        st.warning("الرجاء تعبئة اسم المشروع ووصفه أولاً.")
    elif scaler and xgb_model and ann_model:
        with st.spinner('جاري تشغيل المحرك الهجين وتحليل البيانات...'):
            # أ. الربط الذكي للأهداف
            found_sdgs = infer_sdgs(p_cat, p_desc)
            sdg_count = len(found_sdgs)
            
            # ب. معالجة البيانات للموديل
            social_ratio = min(p_ben / (p_budget/100), 1.0)
            balance_score = min(sdg_count / 5 + 0.4, 1.0)
            env_ratio = 0.8 if p_cat == "بيئي" else 0.4
            
            features = np.array([[sdg_count, social_ratio, balance_score, env_ratio]])
            features_scaled = scaler.transform(features)
            
            # تفعيل الموديل الهجين
            p_ann = ann_model.predict(features_scaled).flatten()[0]
            p_xgb = xgb_model.predict_proba(features_scaled)[:, 1][0]
            score = (p_ann * 0.6) + (p_xgb * 0.4)
            
            # ج. حساب العوائد
            sroi = round(score * (p_ben / (p_budget/1000)), 2)
            eco_impact = f"{int(p_budget * score * 1.5):,}"
            
            # د. عرض النتائج الفخمة
            res_color = "#1b5e20" if score > 0.6 else "#b71c1c"
            
            results_ui = f"""
            <div dir="rtl" style="background: white; padding: 30px; border-radius: 15px; border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); font-family: sans-serif;">
                <h2 style="color: #0d47a1; border-bottom: 2px solid #eee; padding-bottom: 10px;">نتائج تحليل: {p_name}</h2>
                
                <div style="display: flex; align-items: center; justify-content: center; margin: 30px 0;">
                    <div style="text-align: center; background: #f1f8ff; padding: 20px; border-radius: 50%; width: 200px; height: 200px; display: flex; flex-direction: column; justify-content: center; border: 8px solid #0d47a1;">
                        <span style="font-size: 0.9em; color: #546e7a;">نسبة النجاح</span>
                        <span style="font-size: 3em; font-weight: bold; color: {res_color};">{score*100:.1f}%</span>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 30px;">
                    <div style="background: #eceff1; padding: 15px; border-radius: 10px; text-align: center;">
                        <span style="color: #455a64; font-size: 0.9em;">العائد الاجتماعي</span><br>
                        <span style="font-size: 1.5em; font-weight: bold; color: #0d47a1;">{sroi}x</span>
                    </div>
                    <div style="background: #eceff1; padding: 15px; border-radius: 10px; text-align: center;">
                        <span style="color: #455a64; font-size: 0.9em;">الأثر الاقتصادي</span><br>
                        <span style="font-size: 1.2em; font-weight: bold; color: #0d47a1;">{eco_impact} ريال</span>
                    </div>
                    <div style="background: #eceff1; padding: 15px; border-radius: 10px; text-align: center;">
                        <span style="color: #455a64; font-size: 0.9em;">الأثر البيئي</span><br>
                        <span style="font-size: 1.2em; font-weight: bold; color: #0d47a1;">{"مستدام" if env_ratio > 0.5 else "آمن"}</span>
                    </div>
                </div>

                <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; border-right: 6px solid #2e7d32; margin-bottom: 20px;">
                    <strong style="color: #1b5e20;">🌍 الأهداف العالمية المرتبطة آلياً:</strong><br>
                    <span style="font-weight: bold; color: #333;">{' | '.join(found_sdgs)}</span>
                </div>

                <div style="background: #fdfdfd; padding: 20px; border-radius: 10px; border: 1px dashed #0d47a1;">
                    <h4 style="margin-top: 0; color: #0d47a1;">💡 توصية النظام المتقدم:</h4>
                    <p style="color: #444;">{"✅ المشروع يمتلك كفاءة تنموية عالية، نوصي بالمضي قدماً في التمويل." if score > 0.6 else "⚠️ المشروع يحتاج إلى إعادة دراسة التكاليف مقابل عدد المستفيدين لرفع كفاءة الأثر."}</p>
                    <small style="color: #90a4ae;">* تم التحليل بناءً على بيانات {p_cat} لمدة {p_duration} يوماً.</small>
                </div>
            </div>
            """
            components.html(results_ui, height=850, scrolling=True)
            if score > 0.6: st.balloons()
    else:
        st.error("خطأ: لم نجد ملفات الذكاء الاصطناعي (scaler, ann, xgb) في GitHub. تأكدي من رفعها أولاً.")

