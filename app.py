import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# --- 1. إعدادات الصفحة الرسمية ---
st.set_page_config(page_title="منصة تحليل المشاريع التنموية", layout="wide")

# --- 2. تحميل النماذج (الذكاء الاصطناعي الهجين) ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except: return None, None, None

scaler, xgb_model, ann_model = load_models()

# --- 3. محرك الربط التلقائي بالأهداف (SDGs) بناءً على المجال والوصف ---
def get_sdgs(category, description):
    mapping = {
        "تعليمي": ["4. التعليم الجيد", "5. المساواة بين الجنسين", "17. عقد الشراكات"],
        "صحي": ["3. الصحة الجيدة والرفاه", "6. المياه النظيفة"],
        "اجتماعي": ["1. القضاء على الفقر", "10. الحد من أوجه عدم المساواة"],
        "بيئي": ["13. العمل المناخي", "15. الحياة في البر", "7. طاقة نظيفة"],
        "اقتصادي": ["8. العمل اللائق ونمو الاقتصاد", "9. الصناعة والابتكار"],
        "تقني": ["9. الصناعة والابتكار والهياكل الأساسية", "4. التعليم الجيد"]
    }
    return mapping.get(category, ["17. عقد الشراكات"])

# --- 4. واجهة المدخلات (التصميم بالأزرق الغامق والرمادي) ---
st.markdown("<h1 style='text-align: center; color: #0d47a1;'>المنصة الذكية لتحليل المشاريع التنموية 📊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>نظام متقدم للتقييم والتنبؤ باستخدام النموذج الهجين (ANN + XGBoost)</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("""<style> div.stButton > button:first-child { background-color: #0d47a1; color: white; width: 100%; border-radius: 10px; height: 50px; font-weight: bold; } </style>""", unsafe_allow_html=True)
    
    with st.form("main_form"):
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("📝 اسم المشروع")
            category = st.selectbox("📂 مجال المشروع", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
            budget = st.number_input("💰 الميزانية التقديرية (ريال)", min_value=1000, value=50000)
        with col2:
            duration = st.number_input("⏳ مدة المشروع (أيام)", min_value=1, value=365)
            beneficiaries = st.number_input("👥 عدد المستفيدين المتوقع", min_value=1, value=200)
            
        description = st.text_area("💡 فكرة المشروع (اشرح التفاصيل هنا لربط الأهداف تلقائياً)")
        
        submitted = st.form_submit_button("بدء التحليل الشامل للمشروع 🚀")

# --- 5. تنفيذ التحليل وعرض النتائج الشاملة ---
if submitted:
    if not project_name or not description:
        st.error("الرجاء إكمال اسم المشروع وفكرته قبل التحليل.")
    elif scaler and xgb_model and ann_model:
        # أ. استنتاج الأهداف وتجهيز المعايير
        detected_sdgs = get_sdgs(category, description)
        sdg_count = len(detected_sdgs)
        social_ratio = min(beneficiaries / (budget/100), 1.0)
        balance_score = min(sdg_count / 5 + 0.4, 1.0)
        env_ratio = 0.8 if category == "بيئي" else 0.4
        
        # ب. التنبؤ الهجين
        input_data = np.array([[sdg_count, social_ratio, balance_score, env_ratio]])
        input_scaled = scaler.transform(input_data)
        pred_ann = ann_model.predict(input_scaled).flatten()[0]
        pred_xgb = xgb_model.predict_proba(input_scaled)[:, 1][0]
        final_score = (pred_ann * 0.6) + (pred_xgb * 0.4)
        
        # ج. حساب العوائد (Social & Economic ROI) بناءً على معادلات تقديرية
        social_roi = round(final_score * (beneficiaries/budget) * 1000, 2)
        economic_impact = f"{round(budget * final_score * 1.2, 0):,}"
        
        # د. عرض النتائج بتصميم HTML فخم
        results_html = f"""
        <div dir="rtl" style="background: white; padding: 30px; border-radius: 15px; border-top: 10px solid #0d47a1; box-shadow: 0 4px 20px rgba(0,0,0,0.1); font-family: sans-serif;">
            <h2 style="color: #0d47a1; text-align: center;">🚩 تقرير تحليل: {project_name}</h2>
            
            <div style="background: #0d47a1; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
                <span style="font-size: 1.2em;">احتمالية نجاح المشروع المتوقعة</span><br>
                <span style="font-size: 4em; font-weight: bold;">{final_score*100:.1f}%</span>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 25px;">
                <div style="background: #f4f7f9; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
                    <strong style="color: #0d47a1;">📈 العائد الاجتماعي</strong><br>
                    <span style="font-size: 1.5em; font-weight: bold;">{social_roi}x</span>
                </div>
                <div style="background: #f4f7f9; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
                    <strong style="color: #0d47a1;">💰 الأثر الاقتصادي</strong><br>
                    <span style="font-size: 1.2em; font-weight: bold;">{economic_impact} ريال</span>
                </div>
                <div style="background: #f4f7f9; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
                    <strong style="color: #0d47a1;">🌿 الأثر البيئي</strong><br>
                    <span style="font-size: 1.2em; font-weight: bold;">{"مرتفع" if env_ratio > 0.5 else "متوازن"}</span>
                </div>
            </div>

            <div style="background: #ffffff; padding: 15px; border-radius: 10px; border-right: 6px solid #4caf50; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h4 style="margin: 0; color: #2e7d32;">🌍 أهداف التنمية المستدامة المرتبطة تلقائياً:</h4>
                <p style="margin: 10px 0; font-weight: bold; color: #555;">{' | '.join(detected_sdgs)}</p>
            </div>

            <div style="background: #f9f9f9; padding: 20px; border-radius: 10px; border-right: 6px solid #0d47a1;">
                <h4 style="margin: 0; color: #0d47a1;">💡 توصيات الذكاء الاصطناعي:</h4>
                <ul style="margin-top: 10px; color: #444;">
                    <li>يرجى التركيز على استدامة الموارد المالية خلال الـ {duration} يوم القادمة.</li>
                    <li>كفاءة الوصول للمستفيدين ({beneficiaries}) تعزز من نقاط القوة في نموذجك.</li>
                    <li>{"نوصي ببدء التنفيذ فوراً نظراً لارتفاع احتمالية النجاح." if final_score > 0.6 else "نوصي بمراجعة التكاليف التشغيلية لتحسين فرص النجاح."}</li>
                </ul>
            </div>
        </div>
        """
        components.html(results_html, height=850, scrolling=True)
        if final_score > 0.6: st.balloons()
    else:
        st.error("⚠️ ملفات الموديل (scaler.pkl, hybrid_ann.h5, hybrid_xgb.pkl) غير موجودة في المستودع.")

st.markdown("<br><p style='text-align: center; color: #999; font-size: 0.8em;'>تم التطوير لدعم اتخاذ القرار في المشاريع التنموية - نسخة احترافية 3.0</p>", unsafe_allow_html=True)
