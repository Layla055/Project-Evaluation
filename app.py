import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import json

# إعداد الصفحة
st.set_page_config(page_title="المنصة الذكية لتحليل المشاريع", layout="wide")

# وظيفة تحميل النماذج الذكية
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except:
        return None, None, None

scaler, xgb_model, ann_model = load_models()

# تصميم الـ HTML المدمج مع CSS (الأزرق والرمادي)
html_template = """
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f9;
            margin: 0; padding: 20px; text-align: center; color: #333;
        }}
        .container {{
            background: white; padding: 30px; border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            max-width: 900px; margin: auto; border-top: 8px solid #0d47a1;
        }}
        h1 {{ color: #0d47a1; margin-bottom: 10px; }}
        .analysis-box {{
            background: linear-gradient(135deg, #0d47a1, #42a5f5);
            color: white; padding: 30px; border-radius: 15px; margin: 20px 0;
        }}
        .metric {{ font-size: 3.5em; font-weight: bold; }}
        .card {{
            background: #ffffff; padding: 20px; margin: 15px 0;
            border-radius: 10px; border-right: 5px solid #0d47a1;
            text-align: right; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .result-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 15px;
        }}
        .footer-text {{ color: #777; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="80">
        <h1>نتائج التحليل الهجين المتقدم</h1>
        <p>بناءً على نموذج الذكاء الاصطناعي الهجين (ANN + XGBoost)</p>

        <div class="analysis-box">
            <div class="metric">{success_score}%</div>
            <p>احتمالية النجاح المقدرة بدقة الذكاء الاصطناعي</p>
        </div>

        <div class="result-grid">
            <div class="card">
                <h3>📊 كفاءة الموارد</h3>
                <p>تم تحليل الميزانية ({budget} ريال) مقابل عدد المستفيدين ({beneficiaries}) لضمان أعلى عائد اجتماعي.</p>
            </div>
            <div class="card">
                <h3>⚖️ توازن المشروع</h3>
                <p>درجة التوازن المقدرة: {balance_score} استناداً إلى معايير التنمية المستدامة.</p>
            </div>
        </div>

        <div class="card">
            <h3>💡 توصية النظام الذكي</h3>
            <p style="font-size: 1.2em; color: #0d47a1; font-weight: bold;">{recommendation}</p>
        </div>

        <p class="footer-text">تم توليد هذا التقرير آلياً بواسطة المنصة الذكية - مؤتمر إدارة المشاريع</p>
    </div>
</body>
</html>
"""

# واجهة المستخدم في Streamlit (إدخال البيانات)
st.title("⚙️ لوحة تحكم التقييم الذكي")
st.markdown("أدخل بيانات المشروع أدناه ليقوم النموذج الهجين بتحليلها فوراً.")

with st.form("project_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("اسم المشروع التنموي")
        category = st.selectbox("مجال المشروع", ["تعليمي", "صحي", "اجتماعي", "بيئي", "اقتصادي", "تقني"])
        budget = st.number_input("الميزانية التقديرية (ريال)", min_value=1000, value=50000)
    with col2:
        beneficiaries = st.number_input("عدد المستفيدين المتوقع", min_value=1, value=500)
        duration = st.number_input("المدة الزمنية (أيام)", min_value=1, value=365)
        sdg_count = st.slider("عدد أهداف التنمية المستدامة المرتبطة", 1, 17, 5)

    submitted = st.form_submit_button("تشغيل التحليل العميق 🚀")

if submitted:
    if scaler and xgb_model and ann_model:
        # حساب الميزات بناءً على مدخلات المستخدم لمحاكاة البيانات التي تدرب عليها الموديل
        social_ratio = min(beneficiaries / (budget/100), 1.0)
        balance_score = min(sdg_count / 17 + 0.3, 1.0)
        env_ratio = 0.6 if category == "بيئي" else 0.4
        
        # تجهيز البيانات للموديل
        input_data = np.array([[sdg_count, social_ratio, balance_score, env_ratio]])
        input_scaled = scaler.transform(input_data)
        
        # التنبؤ الهجين
        pred_ann = ann_model.predict(input_scaled).flatten()[0]
        pred_xgb = xgb_model.predict_proba(input_scaled)[:, 1][0]
        final_score = (pred_ann * 0.6) + (pred_xgb * 0.4)
        
        # تحديد التوصية
        if final_score > 0.7:
            rec = "✅ المشروع يمتلك مقومات نجاح عالية جداً، نوصي بالتنفيذ الفوري."
        elif final_score > 0.4:
            rec = "⚠️ المشروع واعد ولكن يتطلب تحسين في توزيع الموارد لضمان الاستدامة."
        else:
            rec = "❌ مخاطر المشروع مرتفعة، ينصح بإعادة النظر في خطة التنفيذ والميزانية."

        # عرض الـ HTML المخصص مع النتائج
        full_html = html_template.format(
            success_score=round(final_score * 100, 2),
            budget=budget,
            beneficiaries=beneficiaries,
            balance_score=round(balance_score, 2),
            recommendation=rec
        )
        
        components.html(full_html, height=800, scrolling=True)
    else:
        st.error("خطأ: لم يتم العثور على ملفات الموديل الذكي. تأكدي من رفع scaler.pkl و hybrid_ann.h5 و hybrid_xgb.pkl")
