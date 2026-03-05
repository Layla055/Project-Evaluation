import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="PMD Evaluation Platform", layout="centered")

# تحميل الموديلات
@st.cache_resource
def load_models():
    # تأكدي من رفع الملفات في GitHub في نفس المجلد الرئيسي
    scaler = joblib.load('scaler.pkl')
    xgb_model = joblib.load('hybrid_xgb.pkl')
    ann_model = tf.keras.models.load_model('hybrid_ann.h5')
    return scaler, xgb_model, ann_model

try:
    scaler, xgb_model, ann_model = load_models()
    st.success("✅ تم تحميل محرك الذكاء الاصطناعي بنجاح")
except:
    st.warning("⚠️ جاري إعداد محرك التقييم...")

# واجهة المستخدم
st.title("نظام تقييم المشاريع التنموية (Hybrid AI)")
st.subheader("مؤتمر إدارة المشاريع التنموية PMD")

# مدخلات البيانات التنموية (بناءً على Features الخاصة بكِ)
col1, col2 = st.columns(2)
with col1:
    sdg_count = st.number_input("عدد أهداف التنمية المستدامة (SDG_count)", 0, 17)
    social_ratio = st.slider("نسبة الأثر الاجتماعي (Social_ratio)", 0.0, 1.0)
with col2:
    balance_score = st.slider("درجة التوازن (Balance_score)", 0.0, 1.0)
    env_ratio = st.slider("نسبة الأثر البيئي (Environmental_ratio)", 0.0, 1.0)

if st.button("تحليل المشروع الآن"):
    # تجهيز البيانات للتنبؤ
    input_data = np.array([[sdg_count, social_ratio, balance_score, env_ratio]])
    input_scaled = scaler.transform(input_data)
    
    # التنبؤ الهجين (Hybrid: ANN + XGBoost)
    pred_ann = ann_model.predict(input_scaled).flatten()[0]
    pred_xgb = xgb_model.predict_proba(input_scaled)[:, 1][0]
    
    # دمج النتائج (بناءً على أفضل نسبة ظهرت في الكود الخاص بك 0.6 و 0.4)
    final_score = (pred_ann * 0.6) + (pred_xgb * 0.4)
    
    # عرض النتيجة
    st.markdown("---")
    if final_score > 0.5:
        st.balloons()
        st.success(f"### النتيجة: مشروع ناجح واعد \n **نسبة النجاح المتوقعة: {final_score*100:.2f}%**")
    else:
        st.error(f"### النتيجة: المشروع يتطلب إعادة تقييم \n **نسبة النجاح المتوقعة: {final_score*100:.2f}%**")
