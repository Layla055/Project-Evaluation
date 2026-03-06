import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import os
import re
from pathlib import Path

# محاولة تحميل TensorFlow بشكل آمن
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="المنصة الذكية لتحليل المشاريع التنموية", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. دوال مساعدة للتحقق من الملفات ---
def check_model_files():
    """التحقق من وجود ملفات النماذج"""
    model_files = {
        'scaler.pkl': False,
        'hybrid_xgb.pkl': False,
        'hybrid_ann.h5': False,
        'config.pkl': False
    }
    
    for file in model_files.keys():
        model_files[file] = os.path.exists(file)
    
    return model_files

# --- 3. تحميل النماذج مع معالجة الأخطاء ---
@st.cache_resource
def load_models_safe():
    """تحميل النماذج مع التعامل مع الأخطاء"""
    models = {
        'scaler': None,
        'xgb': None,
        'ann': None,
        'config': None,
        'status': 'no_models'
    }
    
    try:
        files_status = check_model_files()
        
        if files_status['scaler.pkl']:
            models['scaler'] = joblib.load('scaler.pkl')
        
        if files_status['hybrid_xgb.pkl']:
            models['xgb'] = joblib.load('hybrid_xgb.pkl')
        
        if files_status['hybrid_ann.h5'] and TENSORFLOW_AVAILABLE:
            models['ann'] = load_model('hybrid_ann.h5')
        
        if files_status['config.pkl']:
            models['config'] = joblib.load('config.pkl')
        
        if any([models['scaler'], models['xgb'], models['ann']]):
            models['status'] = 'partial_models'
        if models['scaler'] and models['xgb'] and models['ann']:
            models['status'] = 'full_models'
            
    except Exception as e:
        models['status'] = 'error'
    
    return models

# --- 4. قاموس الكلمات المفتاحية ---
SDG_KEYWORDS = {
    1: {'name': 'القضاء على الفقر', 
        'keywords': ['فقر', 'فقراء', 'تمكين', 'دخل', 'مساعدات', 'ضمان', 'تكافل', 'poverty']},
    2: {'name': 'القضاء على الجوع', 
        'keywords': ['جوع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 'hunger']},
    3: {'name': 'الصحة الجيدة', 
        'keywords': ['صحة', 'مستشفى', 'مركز صحي', 'رعاية', 'أمراض', 'علاج', 'health']},
    4: {'name': 'التعليم الجيد', 
        'keywords': ['تعليم', 'مدرسة', 'جامعة', 'طلاب', 'معلمين', 'تدريب', 'education']},
    5: {'name': 'المساواة بين الجنسين', 
        'keywords': ['مساواة', 'نساء', 'فتيات', 'تمكين المرأة', 'gender']},
    6: {'name': 'المياه النظيفة', 
        'keywords': ['مياه', 'صرف صحي', 'تنقية', 'شرب', 'ري', 'water']},
    7: {'name': 'الطاقة النظيفة', 
        'keywords': ['طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'energy']},
    8: {'name': 'العمل اللائق', 
        'keywords': ['عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'employment']},
    9: {'name': 'الصناعة والابتكار', 
        'keywords': ['صناعة', 'ابتكار', 'بنية تحتية', 'طرق', 'مصانع', 'industry']},
    10: {'name': 'الحد من عدم المساواة', 
         'keywords': ['مساواة', 'فئات مهمشة', 'ذوي احتياجات', 'تمكين', 'inequality']},
    11: {'name': 'مدن مستدامة', 
         'keywords': ['مدن', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل', 'cities']},
    12: {'name': 'استهلاك مسؤول', 
         'keywords': ['استهلاك', 'إنتاج', 'استدامة', 'إعادة تدوير', 'consumption']},
    13: {'name': 'العمل المناخي', 
         'keywords': ['مناخ', 'تغير مناخي', 'انبعاثات', 'كربون', 'climate']},
    14: {'name': 'الحياة تحت الماء', 
         'keywords': ['بحار', 'محيطات', 'أسماك', 'سواحل', 'oceans']},
    15: {'name': 'الحياة في البر', 
         'keywords': ['بيئة', 'غابات', 'تنوع', 'حيوانات', 'نباتات', 'environment']},
    16: {'name': 'السلام والعدالة', 
         'keywords': ['سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'peace']},
    17: {'name': 'الشراكات', 
         'keywords': ['شراكات', 'تعاون', 'تمويل', 'منح', 'partnerships']}
}

# --- 5. تصنيف الأهداف ---
SDG_DIMENSIONS = {
    'social': [1, 2, 3, 4, 5, 10, 11, 16],
    'economic': [8, 9, 12, 17],
    'environmental': [6, 7, 13, 14, 15]
}

# --- 6. دوال التحليل ---
def extract_sdgs_from_text(text):
    """استخراج الأهداف من النص"""
    if not text:
        return []
    
    text = text.lower()
    detected_sdgs = []
    
    for sdg_num, sdg_info in SDG_KEYWORDS.items():
        for keyword in sdg_info['keywords']:
            if keyword.lower() in text:
                detected_sdgs.append(sdg_num)
                break
    
    return list(set(detected_sdgs))

def calculate_sdg_metrics(detected_sdgs):
    """حساب المقاييس"""
    if not detected_sdgs:
        return {
            'sdg_count': 0,
            'social_ratio': 0,
            'economic_ratio': 0,
            'environmental_ratio': 0,
            'balance_score': 0,
            'dimensions': {'social': 0, 'economic': 0, 'environmental': 0}
        }
    
    sdg_count = len(detected_sdgs)
    
    social_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['social'])
    economic_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['economic'])
    environmental_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['environmental'])
    
    social_ratio = (social_count / sdg_count) * 100 if sdg_count > 0 else 0
    economic_ratio = (economic_count / sdg_count) * 100 if sdg_count > 0 else 0
    environmental_ratio = (environmental_count / sdg_count) * 100 if sdg_count > 0 else 0
    
    target = 33.33
    deviations = abs(social_ratio - target) + abs(economic_ratio - target) + abs(environmental_ratio - target)
    balance_score = max(0, 100 - (deviations / 2))
    
    return {
        'sdg_count': sdg_count,
        'social_ratio': social_ratio,
        'economic_ratio': economic_ratio,
        'environmental_ratio': environmental_ratio,
        'balance_score': balance_score,
        'dimensions': {
            'social': social_count,
            'economic': economic_count,
            'environmental': environmental_count
        }
    }

def predict_success_fallback(metrics):
    """نسخة احتياطية للتنبؤ"""
    base_score = 0.5
    sdg_bonus = min(metrics['sdg_count'] * 0.05, 0.2)
    balance_bonus = metrics['balance_score'] * 0.002
    
    score = base_score + sdg_bonus + balance_bonus
    return min(score, 0.95)

def get_project_trend(metrics):
    """تحديد توجه المشروع"""
    ratios = {
        'اجتماعي': metrics['social_ratio'],
        'اقتصادي': metrics['economic_ratio'],
        'بيئي': metrics['environmental_ratio']
    }
    max_dimension = max(ratios, key=ratios.get)
    
    if metrics['balance_score'] > 80:
        return "متوازن"
    elif max_dimension == 'اجتماعي':
        return "اجتماعي"
    elif max_dimension == 'اقتصادي':
        return "اقتصادي"
    else:
        return "بيئي"

# --- 7. تحميل النماذج ---
models = load_models_safe()

# --- 8. التصميم الموحد (أبيض فقط) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* الحقول - كلها بيضاء */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 10px 14px !important;
        background: white !important;
        box-shadow: none !important;
    }
    
    /* زر التحليل - أسود فقط */
    .stButton > button {
        background: #0F172A !important;
        color: white !important;
        border-radius: 8px !important;
        height: 48px !important;
        font-weight: 500 !important;
        border: none !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #1E293B !important;
    }
    
    h1 {
        color: #0F172A;
        text-align: center;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 30px;
    }
    
    /* بطاقات النتائج - كلها بيضاء متطابقة */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 20px 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
        height: 100%;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* شارات SDG - بيضاء بحدود */
    .sdg-badge {
        background: white;
        color: #0F172A;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin: 4px;
        display: inline-block;
        font-size: 0.85rem;
        text-align: center;
    }
    
    /* عناوين الأقسام */
    .section-title {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    
    /* مربع الخلاصة - أبيض */
    .summary-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 9. العنوان ---
st.markdown("<h1>المنصة الذكية لتحليل المشاريع التنموية</h1>", unsafe_allow_html=True)

# --- 10. نموذج الإدخال ---
with st.form("analysis_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        p_name = st.text_input("اسم المشروع", placeholder="أدخل اسم المشروع...")
        p_desc = st.text_area("وصف المشروع", height=150, 
                             placeholder="أدخل تفاصيل المشروع هنا...")
    
    with col2:
        p_cat = st.selectbox("المجال", 
                            ["", "تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
        p_budget = st.number_input("الميزانية (SAR)", min_value=0, value=0)
        p_ben = st.number_input("عدد المستفيدين", min_value=0, value=0)
    
    submitted = st.form_submit_button("تحليل المشروع", use_container_width=True)

# --- 11. التحليل والنتائج ---
if submitted:
    if not p_name or not p_desc or not p_cat or p_budget == 0 or p_ben == 0:
        st.error("يرجى إدخال جميع البيانات المطلوبة")
    else:
        with st.spinner("جاري التحليل..."):
            
            # استخراج الأهداف
            full_text = f"{p_name} {p_desc}"
            detected_sdgs = extract_sdgs_from_text(full_text)
            
            # حساب المقاييس
            metrics = calculate_sdg_metrics(detected_sdgs)
            
            # التنبؤ
            if models['status'] == 'full_models':
                features = np.array([[
                    metrics['sdg_count'],
                    metrics['social_ratio'],
                    metrics['balance_score'],
                    metrics['environmental_ratio']
                ]])
                features_scaled = models['scaler'].transform(features)
                ann_prob = models['ann'].predict(features_scaled, verbose=0)[0][0]
                xgb_prob = models['xgb'].predict_proba(features)[0][1]
                
                config = models['config']
                success_prob = (config['weight_ann'] * ann_prob + config['weight_xgb'] * xgb_prob)
                success_pred = 1 if success_prob >= config['threshold'] else 0
            else:
                success_prob = predict_success_fallback(metrics)
                success_pred = 1 if success_prob >= 0.6 else 0
            
            # المؤشرات المالية
            sroi = round(success_prob * (p_ben / (p_budget/1000)) if p_budget > 0 else 0, 2)
            
            # --- عرض النتائج ببطاقات متساوية الحجم ---
            
            # الصف الأول: 3 بطاقات متساوية
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">نسبة النجاح</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #0F172A;">{success_prob*100:.1f}%</div>
                        <div style="margin-top: 8px; color: {'#10B981' if success_pred == 1 else '#EF4444'}; font-weight: 500;">
                            {'ناجح' if success_pred == 1 else 'غير ناجح'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">الأهداف المستخرجة</div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #0F172A;">{metrics['sdg_count']}</div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.9rem;">هدف من أهداف التنمية</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                trend = get_project_trend(metrics)
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">توجه المشروع</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: #0F172A; margin: 10px 0;">{trend}</div>
                        <div style="color: #64748B; font-size: 0.9rem;">{metrics['balance_score']:.1f}% توازن</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # أهداف SDG
            if detected_sdgs:
                st.markdown('<div class="section-title">أهداف التنمية المستدامة</div>', unsafe_allow_html=True)
                
                # عرض الأهداف في 4 أعمدة
                sdg_cols = st.columns(4)
                for i, sdg in enumerate(detected_sdgs):
                    with sdg_cols[i % 4]:
                        st.markdown(f"""
                            <div class="sdg-badge" style="width: 100%;">
                                الهدف {sdg}: {SDG_KEYWORDS[sdg]['name']}
                            </div>
                        """, unsafe_allow_html=True)
            
            # تحليل الأبعاد - 3 بطاقات متساوية
            st.markdown('<div class="section-title">تحليل الأبعاد</div>', unsafe_allow_html=True)
            
            dim_cols = st.columns(3)
            
            with dim_cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">البعد الاجتماعي</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{metrics['dimensions']['social']}</div>
                        <div style="margin-top: 8px; color: #64748B;">{metrics['social_ratio']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with dim_cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">البعد الاقتصادي</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{metrics['dimensions']['economic']}</div>
                        <div style="margin-top: 8px; color: #64748B;">{metrics['economic_ratio']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with dim_cols[2]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem; margin-bottom: 8px;">البعد البيئي</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{metrics['dimensions']['environmental']}</div>
                        <div style="margin-top: 8px; color: #64748B;">{metrics['environmental_ratio']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # الخلاصة - مربع أبيض
            st.markdown(f"""
                <div class="summary-box">
                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 15px; color: #0F172A;">الخلاصة</div>
                    <p style="color: #334155; line-height: 1.6; margin-bottom: 10px;">
                        مشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong> يستهدف <strong>{metrics['sdg_count']}</strong> 
                        من أهداف التنمية المستدامة. نسبة النجاح المتوقعة <strong>{success_prob*100:.1f}%</strong>.
                    </p>
                    <p style="color: #334155; margin-top: 10px;">
                        <strong>التوصية:</strong> {'المشروع واعد ويمكن المضي قدماً' if success_pred == 1 else 'يحتاج المشروع إلى إعادة هيكلة'}
                    </p>
                </div>
            """, unsafe_allow_html=True)

# --- 12. معلومات النماذج (شريط جانبي) ---
with st.sidebar:
    st.markdown("### معلومات النظام")
    
    files_status = check_model_files()
    available_files = [f for f, exists in files_status.items() if exists]
    
    if available_files:
        st.success(f"✓ الملفات المتاحة: {len(available_files)}")
        for file in available_files:
            st.markdown(f"- {file}")
    else:
        st.warning("⚠️ لا توجد نماذج مدربة")
    
    if models['status'] == 'full_models':
        st.success("✓ جميع النماذج تعمل")
    elif models['status'] == 'partial_models':
        st.info("بعض النماذج فقط متاحة")

# --- 13. التذييل ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; padding: 20px;'>المنصة الذكية لتحليل المشاريع التنموية 2024</div>", unsafe_allow_html=True)
