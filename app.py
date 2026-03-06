import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import os
import re
from pathlib import Path
from collections import Counter

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
        'status': 'no_models',
        'files': {}
    }
    
    try:
        models['files'] = check_model_files()
        
        if models['files']['scaler.pkl']:
            models['scaler'] = joblib.load('scaler.pkl')
        
        if models['files']['hybrid_xgb.pkl']:
            models['xgb'] = joblib.load('hybrid_xgb.pkl')
        
        if models['files']['hybrid_ann.h5'] and TENSORFLOW_AVAILABLE:
            models['ann'] = load_model('hybrid_ann.h5')
        
        if models['files']['config.pkl']:
            models['config'] = joblib.load('config.pkl')
        
        model_count = sum([1 for m in [models['scaler'], models['xgb'], models['ann']] if m is not None])
        if model_count == 3:
            models['status'] = 'full_models'
        elif model_count > 0:
            models['status'] = 'partial_models'
            
    except Exception as e:
        models['status'] = f'error: {str(e)}'
    
    return models

# --- 4. قاموس الكلمات المفتاحية ---
SDG_KEYWORDS = {
    1: {'name': 'القضاء على الفقر', 'keywords': ['فقر', 'فقراء', 'تمكين اقتصادي', 'دخل', 'مساعدات', 'ضمان اجتماعي', 'تكافل', 'poverty', 'poor', 'low income']},
    2: {'name': 'القضاء على الجوع', 'keywords': ['جوع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 'hunger', 'food security', 'agriculture']},
    3: {'name': 'الصحة الجيدة', 'keywords': ['صحة', 'مستشفى', 'مركز صحي', 'رعاية صحية', 'أمراض', 'لقاحات', 'أدوية', 'علاج', 'health', 'hospital', 'medical']},
    4: {'name': 'التعليم الجيد', 'keywords': ['تعليم', 'مدرسة', 'جامعة', 'طلاب', 'معلمين', 'مناهج', 'تدريب', 'محو أمية', 'education', 'school', 'training']},
    5: {'name': 'المساواة بين الجنسين', 'keywords': ['مساواة', 'نساء', 'فتيات', 'تمكين المرأة', 'gender equality', 'women', 'girls']},
    6: {'name': 'المياه النظيفة', 'keywords': ['مياه', 'صرف صحي', 'تنقية', 'شرب', 'ري', 'water', 'sanitation']},
    7: {'name': 'الطاقة النظيفة', 'keywords': ['طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'energy', 'solar', 'renewable']},
    8: {'name': 'العمل اللائق', 'keywords': ['عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'employment', 'jobs', 'labor']},
    9: {'name': 'الصناعة والابتكار', 'keywords': ['صناعة', 'ابتكار', 'بنية تحتية', 'طرق', 'مصانع', 'industry', 'innovation', 'infrastructure']},
    10: {'name': 'الحد من عدم المساواة', 'keywords': ['مساواة', 'فئات مهمشة', 'ذوي احتياجات خاصة', 'كبار السن', 'المسنين', 'inequality', 'marginalized', 'disabled', 'elderly']},
    11: {'name': 'مدن مستدامة', 'keywords': ['مدن', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل عام', 'sustainable cities', 'urban planning']},
    12: {'name': 'استهلاك مسؤول', 'keywords': ['استهلاك', 'إنتاج', 'استدامة', 'إعادة تدوير', 'consumption', 'recycling']},
    13: {'name': 'العمل المناخي', 'keywords': ['مناخ', 'تغير مناخي', 'انبعاثات', 'كربون', 'climate', 'emissions']},
    14: {'name': 'الحياة تحت الماء', 'keywords': ['بحار', 'محيطات', 'أسماك', 'سواحل', 'oceans', 'marine']},
    15: {'name': 'الحياة في البر', 'keywords': ['بيئة', 'غابات', 'تنوع أحيائي', 'حيوانات', 'environment', 'forests', 'biodiversity']},
    16: {'name': 'السلام والعدالة', 'keywords': ['سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'peace', 'justice', 'governance']},
    17: {'name': 'الشراكات', 'keywords': ['شراكات', 'تعاون دولي', 'تمويل', 'منح', 'partnerships', 'cooperation']}
}

# --- 5. قواعد ذكية للكلمات المركبة ---
MULTI_SDG_RULES = [
    {'triggers': ['كبار السن', 'المسنين', 'المسنات', 'المتقاعدين'], 'target_sdgs': [3, 10], 'primary': 10},
    {'triggers': ['تعليم كبار السن', 'محو أمية كبار', 'تعليم المسنين'], 'target_sdgs': [4, 10], 'primary': 10},
    {'triggers': ['أطفال الشوارع', 'أطفال بلا مأوى'], 'target_sdgs': [1, 4, 10], 'primary': 10},
    {'triggers': ['تمكين المرأة الريفية', 'تنمية المرأة الريفية'], 'target_sdgs': [5, 8, 10], 'primary': 5},
    {'triggers': ['زراعة عضوية', 'زراعة مستدامة'], 'target_sdgs': [2, 12, 15], 'primary': 2},
    {'triggers': ['طاقة شمسية للمنازل', 'طاقة متجددة للمجتمعات'], 'target_sdgs': [7, 11], 'primary': 7},
    {'triggers': ['صحة الأم والطفل', 'رعاية الحوامل'], 'target_sdgs': [3, 5], 'primary': 3},
    {'triggers': ['تعليم الفتيات', 'تمكين الفتيات'], 'target_sdgs': [4, 5], 'primary': 4}
]

# --- 6. تصنيف الأهداف ---
SDG_DIMENSIONS = {
    'social': [1, 2, 3, 4, 5, 10, 11, 16],
    'economic': [8, 9, 12, 17],
    'environmental': [6, 7, 13, 14, 15]
}

# --- 7. دوال التحليل ---
def extract_sdgs_from_text_advanced(text):
    """استخراج الأهداف من النص بطريقة ذكية"""
    if not text:
        return [], [], []
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    detected_sdgs = []
    matched_keywords = []
    primary_sdgs = []
    
    for rule in MULTI_SDG_RULES:
        for trigger in rule['triggers']:
            if trigger in text:
                detected_sdgs.extend(rule['target_sdgs'])
                matched_keywords.append(trigger)
                if rule['primary'] not in primary_sdgs:
                    primary_sdgs.append(rule['primary'])
                break
    
    for sdg_num, sdg_info in SDG_KEYWORDS.items():
        for keyword in sdg_info['keywords']:
            if keyword in text and sdg_num not in detected_sdgs:
                detected_sdgs.append(sdg_num)
                matched_keywords.append(keyword)
                break
    
    return list(set(detected_sdgs)), list(set(primary_sdgs)), matched_keywords

def calculate_sdg_metrics(detected_sdgs):
    """حساب المقاييس من الأهداف"""
    if not detected_sdgs:
        return {
            'sdg_count': 0, 'social_ratio': 0, 'economic_ratio': 0, 
            'environmental_ratio': 0, 'balance_score': 0,
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
        'dimensions': {'social': social_count, 'economic': economic_count, 'environmental': environmental_count}
    }

def predict_success_fallback(metrics):
    """نسخة احتياطية للتنبؤ"""
    base_score = 0.5
    sdg_bonus = min(metrics['sdg_count'] * 0.05, 0.25)
    balance_bonus = metrics['balance_score'] * 0.002
    return min(base_score + sdg_bonus + balance_bonus, 0.95)

def get_project_trend(metrics):
    """تحديد توجه المشروع"""
    ratios = {'اجتماعي': metrics['social_ratio'], 'اقتصادي': metrics['economic_ratio'], 'بيئي': metrics['environmental_ratio']}
    max_dim = max(ratios, key=ratios.get)
    
    if metrics['balance_score'] > 80:
        return "متوازن"
    return max_dim

# --- 8. نظام التوصيات الذكي (محدث - يركز على الميزانية والثقة) ---
def generate_recommendations(metrics, p_cat, p_budget, p_ben, success_prob):
    """توليد توصيات ذكية تركز على الميزانية وجدوى الاستثمار"""
    
    recommendations = []
    weaknesses = []
    strengths = []
    
    # 1. تحليل الميزانية وكفاءة الإنفاق (الأهم)
    if p_budget > 0 and p_ben > 0:
        cost_per_person = p_budget / p_ben
        sroi = round((p_ben * success_prob) / (p_budget / 1000), 2)
        
        # تحليل الميزانية - هل هي مناسبة؟
        if p_budget < 50000:
            if p_ben < 100:
                strengths.append("✅ ميزانية مناسبة لمشروع صغير")
            else:
                strengths.append("✅ كفاءة عالية - ميزانية منخفضة ومستفيدين كثر")
        elif p_budget > 1000000:  # أكثر من مليون
            if success_prob > 0.7:
                strengths.append("✅ مشروع استراتيجي كبير يستحق الاستثمار")
            else:
                weaknesses.append("⚠️ مخاطرة عالية - ميزانية كبيرة وفرص نجاح متوسطة")
                recommendations.append("💰 **تقييم المخاطر**: الميزانية كبيرة (أكثر من مليون). يُنصح بعمل دراسة جدوى معمقة وتدقيق إضافي قبل الاعتماد")
        
        # تحليل التكلفة لكل مستفيد
        if cost_per_person > 100000:
            weaknesses.append("⚠️ تكلفة مرتفعة جداً لكل مستفيد")
            recommendations.append("💰 **إعادة هيكلة التكاليف**: التكلفة الحالية مرتفعة جداً. هل يمكن تنفيذ المشروع بكفاءة أعلى؟")
        elif cost_per_person > 50000:
            weaknesses.append("⚠️ تكلفة مرتفعة لكل مستفيد")
            recommendations.append("💰 **ترشيد الإنفاق**: خفض التكاليف بنسبة 20% مع الحفاظ على الجودة")
        elif cost_per_person < 10000:
            strengths.append("✅ كفاءة تشغيلية ممتازة")
        
        # تحليل العائد الاجتماعي - مؤشر ثقة للمستثمر
        if sroi > 10:
            strengths.append(f"✅ عائد اجتماعي ممتاز ({sroi}x)")
            recommendations.append("📊 **فرصة استثمارية واعدة**: العائد الاجتماعي مرتفع جداً، يُنصح بتوسيع نطاق المشروع")
        elif sroi < 1:
            weaknesses.append(f"⚠️ عائد اجتماعي منخفض ({sroi}x)")
            recommendations.append("📉 **تحسين الأثر**: العائد على الاستثمار منخفض. ركز على الفئات الأكثر احتياجاً لزيادة الأثر")
    
    # 2. تحليل الثقة للمستثمر (صاحب القرار)
    confidence_level = "منخفضة"
    if success_prob > 0.7 and metrics['balance_score'] > 60 and p_ben > 100:
        confidence_level = "عالية جداً"
        strengths.append("✅ ثقة عالية في نجاح المشروع")
    elif success_prob > 0.5:
        confidence_level = "متوسطة"
    
    # 3. تحليل عدد الأهداف (بدون إجبار على الزيادة)
    if metrics['sdg_count'] == 0:
        weaknesses.append("⚠️ المشروع غير مرتبط بأهداف تنموية واضحة")
        recommendations.append("🎯 **تحديد الأهداف**: المشروع بحاجة لربط بأهداف التنمية المستدامة ليحظى بثقة المستثمرين")
    elif metrics['sdg_count'] > 4:
        strengths.append("✅ المشروع يغطي عدة أهداف")
    
    # 4. توصيات خاصة بثقة المستثمر
    if confidence_level == "منخفضة" and success_prob < 0.5:
        recommendations.append("⚠️ **موقف استثماري**: نسبة المخاطرة عالية. يوصى بإعادة دراسة المشروع أو البحث عن مصادر تمويل غير تقليدية")
    
    # 5. تحليل القطاع
    sector_success_rates = {
        'تعليمي': 0.76,
        'صحي': 0.72,
        'اجتماعي': 0.74,
        'اقتصادي': 0.68,
        'بيئي': 0.65
    }
    
    if p_cat in sector_success_rates:
        sector_rate = sector_success_rates[p_cat]
        if success_prob > sector_rate + 0.1:
            strengths.append(f"✅ أداء متوقع أفضل من متوسط المشاريع {p_cat}ة")
        elif success_prob < sector_rate - 0.1:
            weaknesses.append(f"⚠️ أداء متوقع أقل من متوسط المشاريع {p_cat}ة")
            recommendations.append(f"📚 **الاستفادة من التجارب**: ادرس المشاريع {p_cat}ة الناجحة واستفد من منهجياتها")
    
    return strengths, weaknesses, recommendations, confidence_level

# --- 9. تحميل النماذج ---
models = load_models_safe()

# --- 10. التصميم ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
        box-sizing: border-box;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #F9FAFB;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #E5E7EB !important;
        padding: 12px 16px !important;
        background: white !important;
    }
    
    .stButton > button {
        background: #0F172A !important;
        color: white !important;
        border-radius: 10px !important;
        height: 52px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        margin-top: 10px !important;
    }
    
    .stButton > button:hover {
        background: #1E293B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    h1 {
        color: #0F172A;
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 20px 0 30px 0;
    }
    
    .success-card {
        background: white;
        border-radius: 16px;
        padding: 30px 20px;
        border: 1px solid #E5E7EB;
        text-align: center;
        max-width: 350px;
        margin: 0 auto 30px auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    .success-card .label {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 10px;
    }
    
    .success-card .value {
        font-size: 4rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .success-card .status {
        margin-top: 15px;
        font-weight: 600;
        padding: 6px 20px;
        border-radius: 30px;
        display: inline-block;
    }
    
    .sdg-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }
    
    .sdg-badge {
        background: white;
        color: #1F2937;
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .sdg-badge:hover {
        border-color: #0F172A;
        transform: translateY(-2px);
    }
    
    .sdg-badge.primary {
        background: #F8FAFC;
        border-right: 4px solid #0F172A;
    }
    
    .section-title {
        color: #0F172A;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 40px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
    }
    
    .section-title .count {
        background: #F3F4F6;
        color: #4B5563;
        font-size: 0.9rem;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 12px;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 20px 0;
    }
    
    .info-card {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px;
    }
    
    .info-card .label {
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    
    .info-card .value {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .dimensions-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .dimensions-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        text-align: center;
    }
    
    .dimension-item .title {
        color: #6B7280;
        margin-bottom: 8px;
    }
    
    .dimension-item .number {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #F3F4F6;
        border-radius: 20px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: #0F172A;
        border-radius: 20px;
    }
    
    .confidence-high {
        background: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .confidence-medium {
        background: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .confidence-low {
        background: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .strengths-box {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .weaknesses-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .recommendations-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .strength-item {
        color: #065F46;
        padding: 8px 0;
        border-bottom: 1px solid #A7F3D0;
    }
    
    .weakness-item {
        color: #991B1B;
        padding: 8px 0;
        border-bottom: 1px solid #FECACA;
    }
    
    .recommendation-item {
        color: #1E3A8A;
        padding: 12px 0;
        border-bottom: 1px solid #BFDBFE;
        font-size: 1.05rem;
    }
    
    .recommendation-item:last-child {
        border-bottom: none;
    }
    
    .summary-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 30px;
        margin: 40px 0 20px 0;
    }
    
    .footer {
        text-align: center;
        color: #9CA3AF;
        padding: 30px 0 20px 0;
        border-top: 1px solid #E5E7EB;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 11. العنوان ---
st.markdown("<h1>المنصة الذكية لتحليل المشاريع التنموية</h1>", unsafe_allow_html=True)

# --- 12. نموذج الإدخال ---
with st.form("analysis_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        p_name = st.text_input("اسم المشروع", placeholder="أدخل اسم المشروع...")
        p_desc = st.text_area("وصف المشروع", height=150, 
                             placeholder="اكتب وصفاً لفكرة المشروع هنا...")
    
    with col2:
        p_cat = st.selectbox("المجال", ["", "تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
        p_budget = st.number_input("الميزانية (SAR)", min_value=0, value=0, step=1000)
        p_ben = st.number_input("عدد المستفيدين", min_value=0, value=0, step=100)
    
    submitted = st.form_submit_button("تحليل المشروع", use_container_width=True)

# --- 13. التحليل والنتائج مع التوصيات ---
if submitted:
    if not p_name or not p_desc or not p_cat or p_budget == 0 or p_ben == 0:
        st.error("⚠️ يرجى إدخال جميع البيانات المطلوبة")
    else:
        with st.spinner("🔍 جاري التحليل الذكي..."):
            
            # استخراج الأهداف
            full_text = f"{p_name} {p_desc}"
            detected_sdgs, primary_sdgs, matched_keywords = extract_sdgs_from_text_advanced(full_text)
            
            # حساب المقاييس
            metrics = calculate_sdg_metrics(detected_sdgs)
            
            # التنبؤ باستخدام النموذج
            if models['status'] == 'full_models' and all([models['scaler'], models['xgb'], models['ann']]):
                try:
                    features = np.array([[
                        metrics['sdg_count'],
                        metrics['social_ratio'],
                        metrics['balance_score'],
                        metrics['environmental_ratio']
                    ]])
                    
                    features_scaled = models['scaler'].transform(features)
                    ann_prob = models['ann'].predict(features_scaled, verbose=0)[0][0]
                    xgb_prob = models['xgb'].predict_proba(features)[0][1]
                    
                    weight_ann = models['config'].get('weight_ann', 0.5)
                    weight_xgb = models['config'].get('weight_xgb', 0.5)
                    threshold = models['config'].get('threshold', 0.4)
                    
                    success_prob = (weight_ann * ann_prob + weight_xgb * xgb_prob)
                    success_pred = 1 if success_prob >= threshold else 0
                    
                except Exception as e:
                    success_prob = predict_success_fallback(metrics)
                    success_pred = 1 if success_prob >= 0.6 else 0
            else:
                success_prob = predict_success_fallback(metrics)
                success_pred = 1 if success_prob >= 0.6 else 0
            
            # --- توليد التوصيات الذكية (المحدثة) ---
            strengths, weaknesses, recommendations, confidence_level = generate_recommendations(
                metrics, p_cat, p_budget, p_ben, success_prob
            )
            
            # عرض نسبة النجاح
            status_color = "#10B981" if success_pred == 1 else "#EF4444"
            status_text = "ناجح" if success_pred == 1 else "غير ناجح"
            
            st.markdown(f"""
                <div class="success-card">
                    <div class="label">نسبة نجاح المشروع</div>
                    <div class="value">{success_prob*100:.1f}%</div>
                    <div class="status" style="background: {status_color}15; color: {status_color};">
                        {status_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # مستوى الثقة للمستثمر
            confidence_class = "confidence-high" if confidence_level == "عالية جداً" else "confidence-medium" if confidence_level == "متوسطة" else "confidence-low"
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <span class="{confidence_class}">مستوى الثقة للمستثمر: {confidence_level}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # --- عرض نقاط القوة ---
            if strengths:
                st.markdown("""
                    <div class="strengths-box">
                        <h4 style="color: #065F46; margin-bottom: 15px;">✅ نقاط القوة</h4>
                """, unsafe_allow_html=True)
                
                for strength in strengths:
                    st.markdown(f'<div class="strength-item">✓ {strength}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # --- عرض نقاط الضعف ---
            if weaknesses:
                st.markdown("""
                    <div class="weaknesses-box">
                        <h4 style="color: #991B1B; margin-bottom: 15px;">⚠️ نقاط تحتاج تحسين</h4>
                """, unsafe_allow_html=True)
                
                for weakness in weaknesses:
                    st.markdown(f'<div class="weakness-item">• {weakness}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # --- عرض التوصيات ---
            if recommendations:
                st.markdown("""
                    <div class="recommendations-box">
                        <h4 style="color: #1E3A8A; margin-bottom: 20px;">💡 توصيات للتحسين</h4>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # أهداف التنمية المستدامة
            if detected_sdgs:
                st.markdown(f"""
                    <div class="section-title">
                        أهداف التنمية المستدامة المرتبطة بالمشروع
                        <span class="count">{len(detected_sdgs)} أهداف</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="sdg-grid">', unsafe_allow_html=True)
                
                for sdg in detected_sdgs:
                    primary_class = "primary" if sdg in primary_sdgs else ""
                    star = "⭐ " if sdg in primary_sdgs else ""
                    
                    st.markdown(f"""
                        <div class="sdg-badge {primary_class}">
                            {star}الهدف {sdg}: {SDG_KEYWORDS[sdg]['name']}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # معلومات إضافية
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">توجه المشروع</div>
                        <div class="value">{get_project_trend(metrics)}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">عدد الأهداف</div>
                        <div class="value">{metrics['sdg_count']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">درجة التوازن</div>
                        <div class="value">{metrics['balance_score']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # توزيع الأهداف
            st.markdown("""
                <div class="dimensions-box">
                    <div style="font-weight: 600; color: #0F172A; margin-bottom: 20px;">توزيع الأهداف على الأبعاد</div>
                    <div class="dimensions-grid">
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                        <div class="dimension-item">
                            <div class="title">اجتماعي</div>
                            <div class="number">{metrics['dimensions']['social']}</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {metrics['social_ratio']}%;"></div></div>
                        </div>
                        <div class="dimension-item">
                            <div class="title">اقتصادي</div>
                            <div class="number">{metrics['dimensions']['economic']}</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {metrics['economic_ratio']}%;"></div></div>
                        </div>
                        <div class="dimension-item">
                            <div class="title">بيئي</div>
                            <div class="number">{metrics['dimensions']['environmental']}</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {metrics['environmental_ratio']}%;"></div></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # مؤشرات مالية
            if p_budget > 0 and p_ben > 0:
                cost_per_person = p_budget / p_ben
                sroi = round((p_ben * success_prob) / (p_budget / 1000), 2)
                
                st.markdown("""
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 20px 0;">
                """, unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="label">التكلفة لكل مستفيد</div>
                            <div class="value">{cost_per_person:,.0f}</div>
                            <span style="color: #9CA3AF;">ريال</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="label">العائد الاجتماعي</div>
                            <div class="value">{sroi}</div>
                            <span style="color: #9CA3AF;">×</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # الخلاصة مع توصية للمستثمر
            investment_advice = ""
            if confidence_level == "عالية جداً" and success_prob > 0.7:
                investment_advice = "✅ فرصة استثمارية واعدة - يوصى بالتمويل"
            elif confidence_level == "متوسطة" and success_prob > 0.5:
                investment_advice = "⚠️ استثمار متوسط المخاطرة - يوصى بتمويل مشروط بمتابعة"
            else:
                investment_advice = "❌ مخاطرة عالية - يوصى بإعادة الدراسة قبل التمويل"
            
            st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-title">📋 خلاصة التحليل وقرار المستثمر</div>
                    <div class="summary-text">
                        <p>مشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong>، 
                        بميزانية <strong>{p_budget:,.0f} ريال</strong> لـ <strong>{p_ben:,} مستفيد</strong>.</p>
                        <p>نسبة النجاح المتوقعة <strong>{success_prob*100:.1f}%</strong>، 
                        ومستوى الثقة للمستثمر <strong>{confidence_level}</strong>.</p>
                        <div style="margin-top: 15px; padding: 15px; background: #F3F4F6; border-radius: 8px;">
                            <strong>توصية استثمارية:</strong> {investment_advice}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 14. التذييل ---
st.markdown('<div class="footer">المنصة الذكية لتحليل المشاريع التنموية 2024</div>', unsafe_allow_html=True)
