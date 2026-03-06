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

# --- 4. قاموس الكلمات المفتاحية الموسع مع أوزان وأهداف متعددة ---
SDG_KEYWORDS = {
    1: {
        'name': 'القضاء على الفقر',
        'keywords': [
            'فقر', 'فقراء', 'تمكين اقتصادي', 'دخل', 'مساعدات', 'ضمان اجتماعي', 'تكافل', 
            'poverty', 'poor', 'low income'
        ]
    },
    2: {
        'name': 'القضاء على الجوع',
        'keywords': [
            'جوع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 
            'hunger', 'food security', 'agriculture'
        ]
    },
    3: {
        'name': 'الصحة الجيدة',
        'keywords': [
            'صحة', 'مستشفى', 'مركز صحي', 'رعاية صحية', 'أمراض', 'لقاحات', 'أدوية', 'علاج',
            'صحة عامة', 'صحة الأم والطفل', 'الرعاية الأولية', 'الطوارئ', 'الإسعاف', 'عيادات',
            'health', 'hospital', 'clinic', 'medical', 'healthcare'
        ]
    },
    4: {
        'name': 'التعليم الجيد',
        'keywords': [
            'تعليم', 'مدرسة', 'جامعة', 'طلاب', 'معلمين', 'مناهج', 'تدريب', 'محو أمية',
            'التعليم الأساسي', 'التعليم الثانوي', 'التعليم العالي', 'رياض أطفال', 'حضانات',
            'التعليم الفني', 'التعليم المهني', 'مراكز التدريب', 'تنمية المهارات',
            'education', 'school', 'university', 'students', 'teachers', 'training'
        ]
    },
    5: {
        'name': 'المساواة بين الجنسين',
        'keywords': [
            'مساواة', 'نساء', 'فتيات', 'تمكين المرأة', 'عنف ضد المرأة', 'حقوق المرأة',
            'المرأة الريفية', 'المرأة العاملة', 'القيادة النسائية', 'ريادة الأعمال النسائية',
            'gender equality', 'women', 'girls', 'female empowerment'
        ]
    },
    6: {
        'name': 'المياه النظيفة',
        'keywords': [
            'مياه', 'صرف صحي', 'محطات تنقية', 'شرب', 'ري', 'سدود', 'آبار',
            'مياه شرب نظيفة', 'تحلية المياه', 'شبكات المياه', 'معالجة المياه',
            'water', 'clean water', 'sanitation', 'sewage', 'irrigation'
        ]
    },
    7: {
        'name': 'الطاقة النظيفة',
        'keywords': [
            'طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'شبكة كهرباء', 'محطات توليد',
            'الطاقة الشمسية', 'الألواح الشمسية', 'توربينات الرياح', 'الطاقة النووية',
            'energy', 'electricity', 'solar', 'renewable', 'clean energy'
        ]
    },
    8: {
        'name': 'العمل اللائق',
        'keywords': [
            'عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'بطالة', 'مهارات مهنية',
            'العمل اللائق', 'ظروف العمل', 'حقوق العمال', 'العمال', 'الموظفين',
            'employment', 'jobs', 'work', 'labor', 'decent work'
        ]
    },
    9: {
        'name': 'الصناعة والابتكار',
        'keywords': [
            'صناعة', 'ابتكار', 'بنية تحتية', 'طرق', 'جسور', 'مصانع', 'تكنولوجيا',
            'القطاع الصناعي', 'المدن الصناعية', 'التصنيع', 'البحث والتطوير',
            'industry', 'innovation', 'infrastructure', 'technology'
        ]
    },
    10: {
        'name': 'الحد من عدم المساواة',
        'keywords': [
            'مساواة', 'شمولية', 'فئات مهمشة', 'ذوي احتياجات خاصة', 'تمكين',
            'المناطق المهمشة', 'الأرياف', 'المناطق النائية', 'اللاجئين', 'النازحين', 'المهاجرين',
            'ذوو الإعاقة', 'المعاقين', 'المكفوفين', 'الصم', 'ذوي الهمم',
            'كبار السن', 'المسنين', 'المسنات', 'المتقاعدين', 'الأحداث', 'الأطفال',
            'inequality', 'inclusion', 'marginalized', 'disabled', 'refugees', 'elderly'
        ]
    },
    11: {
        'name': 'مدن مستدامة',
        'keywords': [
            'مدن', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل عام', 'بنية تحتية حضرية',
            'المدن المستدامة', 'المجتمعات المستدامة', 'المدن الذكية', 'التخطيط العمراني',
            'الإسكان الميسر', 'الإسكان الاجتماعي', 'الحدائق العامة', 'المساحات الخضراء',
            'sustainable cities', 'urban planning', 'housing', 'public transport'
        ]
    },
    12: {
        'name': 'استهلاك مسؤول',
        'keywords': [
            'استهلاك', 'إنتاج', 'استدامة', 'كفاءة موارد', 'إعادة تدوير',
            'الاستهلاك المستدام', 'الإنتاج المستدام', 'ترشيد الاستهلاك',
            'إعادة التدوير', 'تدوير المخلفات', 'الاقتصاد الدائري',
            'responsible consumption', 'recycling', 'circular economy'
        ]
    },
    13: {
        'name': 'العمل المناخي',
        'keywords': [
            'مناخ', 'تغير مناخي', 'انبعاثات', 'احتباس حراري', 'كربون',
            'التغير المناخي', 'الاحتباس الحراري', 'غازات الدفيئة', 'انبعاثات الكربون',
            'الحياد الكربوني', 'خفض الانبعاثات', 'التكيف مع التغير المناخي',
            'climate', 'climate change', 'emissions', 'carbon', 'global warming'
        ]
    },
    14: {
        'name': 'الحياة تحت الماء',
        'keywords': [
            'بحار', 'محيطات', 'أسماك', 'سواحل', 'ثروة بحرية', 'صيد',
            'الحياة البحرية', 'الكائنات البحرية', 'الشعاب المرجانية', 'الاستزراع السمكي',
            'oceans', 'seas', 'marine life', 'fisheries', 'coral reefs'
        ]
    },
    15: {
        'name': 'الحياة في البر',
        'keywords': [
            'بيئة', 'غابات', 'تنوع أحيائي', 'محيات طبيعية', 'حيوانات', 'نباتات',
            'المحميات الطبيعية', 'الحياة البرية', 'الحيوانات البرية', 'التنوع الحيوي',
            'مكافحة التصحر', 'إعادة التشجير', 'الاستدامة البيئية',
            'environment', 'forests', 'biodiversity', 'wildlife', 'conservation'
        ]
    },
    16: {
        'name': 'السلام والعدالة',
        'keywords': [
            'سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'سيادة القانون',
            'الأمن والسلام', 'الاستقرار', 'مكافحة الفساد', 'الشفافية', 'المساءلة',
            'حقوق الإنسان', 'الحريات العامة', 'الحقوق المدنية',
            'peace', 'justice', 'institutions', 'governance', 'rule of law'
        ]
    },
    17: {
        'name': 'الشراكات',
        'keywords': [
            'شراكات', 'تعاون دولي', 'تمويل', 'منح', 'قروض', 'مساعدات دولية',
            'التعاون المشترك', 'المنظمات الدولية', 'المانحين', 'الجهات المانحة',
            'partnerships', 'international cooperation', 'funding', 'grants'
        ]
    }
}

# --- 5. قواعد ذكية للكلمات التي ترتبط بأهداف متعددة ---
MULTI_SDG_RULES = [
    {
        'triggers': ['كبار السن', 'المسنين', 'المسنات', 'المتقاعدين'],
        'target_sdgs': [3, 10],  # الصحة الجيدة + الحد من عدم المساواة
        'primary': 10  # الهدف الأساسي (الحد من عدم المساواة)
    },
    {
        'triggers': ['تعليم كبار السن', 'محو أمية كبار', 'تعليم المسنين'],
        'target_sdgs': [4, 10],  # التعليم الجيد + الحد من عدم المساواة
        'primary': 10
    },
    {
        'triggers': ['أطفال الشوارع', 'أطفال بلا مأوى'],
        'target_sdgs': [1, 4, 10],  # الفقر + التعليم + عدم المساواة
        'primary': 10
    },
    {
        'triggers': ['تمكين المرأة الريفية', 'تنمية المرأة الريفية'],
        'target_sdgs': [5, 8, 10],  # مساواة + عمل لائق + عدم مساواة
        'primary': 5
    },
    {
        'triggers': ['زراعة عضوية', 'زراعة مستدامة'],
        'target_sdgs': [2, 12, 15],  # جوع + استهلاك مسؤول + حياة في البر
        'primary': 2
    },
    {
        'triggers': ['طاقة شمسية للمنازل', 'طاقة متجددة للمجتمعات'],
        'target_sdgs': [7, 11],  # طاقة نظيفة + مدن مستدامة
        'primary': 7
    },
    {
        'triggers': ['مشاريع صغيرة للنساء', 'تمويل أصغر للنساء'],
        'target_sdgs': [5, 8, 10],  # مساواة + عمل لائق + عدم مساواة
        'primary': 5
    },
    {
        'triggers': ['صحة الأم والطفل', 'رعاية الحوامل'],
        'target_sdgs': [3, 5],  # صحة + مساواة
        'primary': 3
    },
    {
        'triggers': ['تعليم الفتيات', 'تمكين الفتيات'],
        'target_sdgs': [4, 5],  # تعليم + مساواة
        'primary': 4
    },
    {
        'triggers': ['مياه نظيفة للمجتمعات الريفية'],
        'target_sdgs': [6, 10],  # مياه + عدم مساواة
        'primary': 6
    }
]

# --- 6. تصنيف الأهداف ---
SDG_DIMENSIONS = {
    'social': [1, 2, 3, 4, 5, 10, 11, 16],
    'economic': [8, 9, 12, 17],
    'environmental': [6, 7, 13, 14, 15]
}

# --- 7. دوال التحليل الذكية ---
def extract_sdgs_from_text_advanced(text):
    """
    استخراج أهداف التنمية المستدامة من النص بطريقة ذكية
    تدعم استخلاص أهداف متعددة من نفس العبارة
    """
    if not text:
        return [], {}, []
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    detected_sdgs = []
    matched_keywords = {i: [] for i in range(1, 18)}
    primary_sdgs = []
    
    # 1. البحث عن القواعد الخاصة (Multi-SDG)
    for rule in MULTI_SDG_RULES:
        for trigger in rule['triggers']:
            if trigger in text:
                # إضافة جميع الأهداف المرتبطة
                for sdg in rule['target_sdgs']:
                    detected_sdgs.append(sdg)
                    matched_keywords[sdg].append(trigger)
                
                # تسجيل الهدف الأساسي
                if rule['primary'] not in primary_sdgs:
                    primary_sdgs.append(rule['primary'])
                break
    
    # 2. البحث في الكلمات المفتاحية العادية
    for sdg_num, sdg_info in SDG_KEYWORDS.items():
        for keyword in sdg_info['keywords']:
            if keyword in text:
                detected_sdgs.append(sdg_num)
                matched_keywords[sdg_num].append(keyword)
                break
    
    # 3. إزالة التكرارات مع الحفاظ على الترتيب
    unique_sdgs = []
    for sdg in detected_sdgs:
        if sdg not in unique_sdgs:
            unique_sdgs.append(sdg)
    
    return unique_sdgs, matched_keywords, primary_sdgs

def calculate_primary_score(sdg, primary_sdgs):
    """حساب درجة أهمية الهدف (أساسي أو ثانوي)"""
    if sdg in primary_sdgs:
        return "⭐ أساسي"
    return "ثانوي"

def calculate_sdg_metrics(detected_sdgs):
    """حساب المقاييس من الأهداف المستخرجة"""
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
    sdg_bonus = min(metrics['sdg_count'] * 0.05, 0.25)
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

# --- 8. تحميل النماذج ---
models = load_models_safe()

# --- 9. التصميم الاحترافي ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
        box-sizing: border-box;
    }
    
    .stApp {
        background-color: #F9FAFB;
    }
    
    /* تحسين شكل الحقول */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #E5E7EB !important;
        padding: 12px 16px !important;
        background: white !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0F172A !important;
        box-shadow: 0 0 0 2px rgba(15,23,42,0.1) !important;
    }
    
    /* زر التحليل */
    .stButton > button {
        background: #0F172A !important;
        color: white !important;
        border-radius: 10px !important;
        height: 52px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        background: #1E293B !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    /* العنوان الرئيسي */
    h1 {
        color: #0F172A;
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 20px 0 30px 0;
        letter-spacing: -0.5px;
    }
    
    /* بطاقة نسبة النجاح */
    .success-card {
        background: white;
        border-radius: 16px;
        padding: 30px 20px;
        border: 1px solid #E5E7EB;
        text-align: center;
        max-width: 350px;
        margin: 0 auto 30px auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .success-card .label {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 10px;
        font-weight: 500;
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
        font-size: 1.1rem;
        padding: 6px 16px;
        border-radius: 30px;
        display: inline-block;
    }
    
    /* شارات SDG */
    .sdg-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 12px;
        margin-top: 15px;
    }
    
    .sdg-badge {
        background: white;
        color: #1F2937;
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        font-size: 0.95rem;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .sdg-badge:hover {
        border-color: #0F172A;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .sdg-badge.primary {
        background: #F8FAFC;
        border-right: 4px solid #0F172A;
    }
    
    /* عناوين الأقسام */
    .section-title {
        color: #0F172A;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 40px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
        display: flex;
        align-items: center;
    }
    
    .section-title .count {
        background: #F3F4F6;
        color: #4B5563;
        font-size: 0.9rem;
        padding: 4px 10px;
        border-radius: 20px;
        margin-right: 12px;
        font-weight: 500;
    }
    
    /* بطاقات المعلومات */
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
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    
    .info-card .label {
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .info-card .value {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .info-card .unit {
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-right: 5px;
    }
    
    /* مربع الأبعاد */
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
        font-size: 0.95rem;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .dimension-item .number {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .dimension-item .percentage {
        color: #9CA3AF;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* شريط التقدم */
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
        transition: width 0.3s ease;
    }
    
    /* مربع الخلاصة */
    .summary-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 30px;
        margin: 40px 0 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .summary-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 15px;
    }
    
    .summary-text {
        color: #4B5563;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    .summary-divider {
        height: 1px;
        background: #E5E7EB;
        margin: 20px 0;
    }
    
    .recommendation {
        font-weight: 600;
        color: #0F172A;
        font-size: 1.05rem;
    }
    
    /* الشريط الجانبي */
    .css-1d391kg {
        background-color: white;
    }
    
    /* التذييل */
    .footer {
        text-align: center;
        color: #9CA3AF;
        padding: 30px 0 20px 0;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 10. العنوان ---
st.markdown("<h1>المنصة الذكية لتحليل المشاريع التنموية</h1>", unsafe_allow_html=True)

# --- 11. نموذج الإدخال ---
with st.form("analysis_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        p_name = st.text_input("اسم المشروع", placeholder="أدخل اسم المشروع...")
        p_desc = st.text_area("وصف المشروع", height=150, 
                             placeholder="اكتب وصفاً لفكرة المشروع هنا...")
    
    with col2:
        p_cat = st.selectbox("المجال", 
                            ["", "تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
        p_budget = st.number_input("الميزانية (SAR)", min_value=0, value=0)
        p_ben = st.number_input("عدد المستفيدين", min_value=0, value=0)
    
    submitted = st.form_submit_button("تحليل المشروع", use_container_width=True)

# --- 12. التحليل والنتائج ---
if submitted:
    if not p_name or not p_desc or not p_cat or p_budget == 0 or p_ben == 0:
        st.error("⚠️ يرجى إدخال جميع البيانات المطلوبة")
    else:
        with st.spinner("🔍 جاري التحليل الذكي..."):
            
            # استخراج الأهداف بطريقة متقدمة
            full_text = f"{p_name} {p_desc}"
            detected_sdgs, matched_keywords, primary_sdgs = extract_sdgs_from_text_advanced(full_text)
            
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
            
            # عرض نسبة النجاح
            status_color = "#10B981" if success_pred == 1 else "#EF4444"
            status_text = "ناجح" if success_pred == 1 else "غير ناجح"
            
            st.markdown(f"""
                <div class="success-card">
                    <div class="label">نسبة نجاح المشروع</div>
                    <div class="value">{success_prob*100:.1f}%</div>
                    <div class="status" style="background: {status_color}10; color: {status_color};">
                        {status_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # أهداف التنمية المستدامة المرتبطة بالمشروع
            if detected_sdgs:
                st.markdown(f"""
                    <div class="section-title">
                        أهداف التنمية المستدامة المرتبطة بالمشروع
                        <span class="count">{len(detected_sdgs)} أهداف</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # عرض الأهداف في شبكة
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
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 40px; background: white; border-radius: 12px; border: 1px dashed #E5E7EB; margin: 20px 0;">
                        <p style="color: #9CA3AF;">لم يتم العثور على أهداف مرتبطة بالمشروع</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # معلومات إضافية
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">توجه المشروع</div>
                        <div class="value">{get_project_trend(metrics)}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">عدد الأهداف</div>
                        <div class="value">{metrics['sdg_count']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_info3:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="label">درجة التوازن</div>
                        <div class="value">{metrics['balance_score']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # توزيع الأهداف على الأبعاد
            st.markdown("""
                <div class="dimensions-box">
                    <div style="font-weight: 600; color: #0F172A; margin-bottom: 20px;">توزيع الأهداف على الأبعاد</div>
                    <div class="dimensions-grid">
            """, unsafe_allow_html=True)
            
            # حساب النسب المئوية
            social_percent = metrics['social_ratio']
            economic_percent = metrics['economic_ratio']
            environmental_percent = metrics['environmental_ratio']
            
            st.markdown(f"""
                        <div class="dimension-item">
                            <div class="title">اجتماعي</div>
                            <div class="number">{metrics['dimensions']['social']}</div>
                            <div class="percentage">{social_percent:.1f}%</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {social_percent}%;"></div></div>
                        </div>
                        <div class="dimension-item">
                            <div class="title">اقتصادي</div>
                            <div class="number">{metrics['dimensions']['economic']}</div>
                            <div class="percentage">{economic_percent:.1f}%</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {economic_percent}%;"></div></div>
                        </div>
                        <div class="dimension-item">
                            <div class="title">بيئي</div>
                            <div class="number">{metrics['dimensions']['environmental']}</div>
                            <div class="percentage">{environmental_percent:.1f}%</div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {environmental_percent}%;"></div></div>
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
                            <span class="unit">ريال</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="label">العائد الاجتماعي المتوقع</div>
                            <div class="value">{sroi}</div>
                            <span class="unit">×</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # الخلاصة
            st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-title">📋 خلاصة التحليل</div>
                    <div class="summary-text">
                        <p style="margin-bottom: 15px;">
                            مشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong>، 
                            يستهدف <strong>{metrics['sdg_count']}</strong> من أهداف التنمية المستدامة 
                            مع تركيز {get_project_trend(metrics)}.
                        </p>
                        <p style="margin-bottom: 15px;">
                            نسبة النجاح المتوقعة <strong>{success_prob*100:.1f}%</strong> 
                            بناءً على تحليل {metrics['sdg_count']} أهداف ودرجة توازن {metrics['balance_score']:.1f}%.
                        </p>
                        <div class="summary-divider"></div>
                        <p class="recommendation">
                            {'✅ يوصى بالمضي قدماً في المشروع' if success_pred == 1 else '⚠️ يوصى بإعادة هيكلة المشروع لتحسين فرص النجاح'}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 13. الشريط الجانبي ---
with st.sidebar:
    st.markdown("### ℹ️ معلومات النظام")
    
    files_status = check_model_files()
    available_files = [f for f, exists in files_status.items() if exists]
    
    if available_files:
        st.success(f"✅ النماذج المتاحة: {len(available_files)}")
    else:
        st.warning("⚠️ لا توجد نماذج مدربة")
    
    st.markdown("---")
    st.markdown("### ✨ ميزات التحليل")
    st.markdown("""
    • استخلاص أهداف متعددة ذكياً
    • تحديد الأهداف الأساسية ⭐
    • قواعد خاصة للكلمات المركبة
    • تحليل التوازن بين الأبعاد
    • مؤشرات مالية متقدمة
    """)
    
    st.markdown("---")
    st.markdown("### 📊 دليل الأهداف")
    st.markdown("""
    **اجتماعي:** 1,2,3,4,5,10,11,16  
    **اقتصادي:** 8,9,12,17  
    **بيئي:** 6,7,13,14,15
    """)

# --- 14. التذييل ---
st.markdown('<div class="footer">المنصة الذكية لتحليل المشاريع التنموية 2024</div>', unsafe_allow_html=True)
