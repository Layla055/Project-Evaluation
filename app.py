import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image  # سنحتاج هذه المكتبة لعرض الصورة

# --- 1. إعدادات الصفحة والهوية البصرية ---
st.set_page_config(page_title="نظام تقييم المشاريع الذكي", layout="centered")

# تطبيق التنسيق والألوان (أبيض، أزرق داكن، رمادي)
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    h1 {
        color: #0d47a1; /* أزرق داكن للعنوان */
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #1976d2; /* أزرق متوسط للأزرار */
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
    p, label {
        color: #333333; /* رمادي داكن للقراءة */
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. إضافة الشعار (استبدلي 'logo.png' باسم ملف شعارك) ---
try:
    image = Image.open('logo.png')
    st.image(image, caption='المؤتمر الدولي لإدارة المشاريع التنموية', use_container_width=True) # عرض الشعار مع تعليق
except FileNotFoundError:
    st.warning("⚠️ لم يتم العثور على ملف الشعار 'logo.png'. تأكدي من رفعه إلى GitHub.")

# --- 3. وظيفة تحميل النماذج الذكية ---
@st.cache_resource
def load_smart_models():
    try:
        scaler = joblib.load('scaler.pkl')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        ann_model = tf.keras.models.load_model('hybrid_ann.h5')
        return scaler, xgb_model, ann_model
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النماذج: {e}")
        return None, None, None

scaler, xgb_model, ann_model = load_smart_models()

# --- 4. واجهة المستخدم ---
st.title("منصة تقييم أثر المشاريع التنموية 📊")
st.write("استخدم هذا النظام الذكي المبني على النموذج الهجين (Hybrid AI) لتقييم فرص نجاح مشروعك التنموي بناءً على المعايير العالمية.")

st.markdown("---")

# الحاوية الخاصة بمدخلات البيانات
with st.container():
    st.subheader("📝 إدخال بيانات المشروع")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sdg_count = st.number_input("عدد أهداف التنمية المستدامة المرتبطة", min_value=0, max_value=17, value=5)
        social_ratio = st.slider("مؤشر الأثر الاجتماعي", 0.0, 1.0, 0.5)
        
    with col2:
        balance_score = st.slider("درجة توازن الموارد", 0.0, 1.0, 0.5)
        env_ratio = st.slider("مؤشر الأثر البيئي", 0.0, 1.0, 0.5)

    st.markdown("###")
    
    if st.button("بدء التقييم الذكي للمشروع"):
        if scaler and xgb_model and ann_model:
            # تجهيز البيانات
            input_features = np.array([[sdg_count, social_ratio, balance_score, env_ratio]])
            input_scaled = scaler.transform(input_features)
            
            # التنبؤ الهجين
            pred_ann = ann_model.predict(input_scaled).flatten()[0]
            pred_xgb = xgb_model.predict_proba(input_scaled)[:, 1][0]
            
            final_success_score = (pred_ann * 0.6) + (pred_xgb * 0.4)
            final_percentage = final_success_score * 100
            
            # عرض النتائج
            st.markdown("---")
            if final_success_score >= 0.5:
                st.balloons()
                st.success(f"### النتيجة: مشروع واعد ذو جدوى عالية \n **نسبة النجاح المتوقعة: {final_percentage:.2f}%**")
                st.info("نوصي بالمضي قدماً في تنفيذ المشروع.")
            else:
                st.error(f"### النتيجة: مخاطر عالية / يتطلب مراجعة \n **نسبة النجاح المتوقعة: {final_percentage:.2f}%**")
                st.warning("يُنصح بإعادة دراسة خطة المخاطر.")
        else:
            st.error("تعذر تشغيل نظام التقييم. تأكدي من رفع كافة ملفات الموديل بنجاح.")

# تذييل الصفحة
st.markdown("---")
st.caption("تم تطوير هذا النظام باستخدام تقنيات الذكاء الاصطناعي الهجين.")
