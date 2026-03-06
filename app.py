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
        return [], {}
    
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
    sdg_bonus = min(metrics['sdg_count'] * 0.05, 0.25)  # زيادة المكافأة
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

# --- 9. التصميم المبسط ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 10px 14px !important;
        background: white !important;
    }
    
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
    
    .success-card {
        background: white;
        border-radius: 8px;
        padding: 30px 20px;
        border: 1px solid #E2E8F0;
        text-align: center;
        max-width: 300px;
        margin: 0 auto 30px auto;
    }
    
    .sdg-badge {
        background: white;
        color: #0F172A;
        padding: 12px 15px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin: 5px;
        display: inline-block;
        font-size: 0.95rem;
        width: 100%;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .sdg-badge.primary {
        border-right: 4px solid #0F172A;
        background: #F8FAFC;
    }
    
    .section-title {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    
    .summary-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 25px;
        margin-top: 30px;
    }
    
    .primary-tag {
        display: inline-block;
        background: #0F172A;
        color: white;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 5px;
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
                             placeholder="أدخل تفاصيل المشروع هنا... مثال: تعليم كبار السن في المناطق الريفية")
    
    with col2:
        p_cat = st.selectbox("المجال", 
                            ["", "تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
        p_budget = st.number_input("الميزانية (SAR)", min_value=0, value=0)
        p_ben = st.number_input("عدد المستفيدين", min_value=0, value=0)
    
    submitted = st.form_submit_button("تحليل المشروع", use_container_width=True)

# --- 12. التحليل والنتائج ---
if submitted:
    if not p_name or not p_desc or not p_cat or p_budget == 0 or p_ben == 0:
        st.error("يرجى إدخال جميع البيانات المطلوبة")
    else:
        with st.spinner("جاري التحليل الذكي..."):
            
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
            st.markdown(f"""
                <div class="success-card">
                    <div style="color: #64748B; font-size: 1rem; margin-bottom: 10px;">نسبة نجاح المشروع</div>
                    <div style="font-size: 4rem; font-weight: 700; color: #0F172A; line-height: 1.2;">{success_prob*100:.1f}%</div>
                    <div style="margin-top: 15px; color: {'#10B981' if success_pred == 1 else '#EF4444'}; font-weight: 500; font-size: 1.1rem;">
                        {'مشروع ناجح' if success_pred == 1 else 'غير ناجح'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # أهداف التنمية المستدامة المرتبطة بالمشروع
            st.markdown('<div class="section-title">أهداف التنمية المستدامة المرتبطة بالمشروع</div>', unsafe_allow_html=True)
            
            if detected_sdgs:
                # عرض عدد الأهداف المكتشفة
                st.markdown(f"""
                    <p style="color: #64748B; margin-bottom: 15px;">
                        تم استخلاص <strong>{len(detected_sdgs)} أهداف</strong> من وصف المشروع
                    </p>
                """, unsafe_allow_html=True)
                
                # عرض الأهداف في 3 أعمدة مع تمييز الأساسي
                sdg_cols = st.columns(3)
                for i, sdg in enumerate(detected_sdgs):
                    primary_class = "primary" if sdg in primary_sdgs else ""
                    primary_text = " ⭐ أساسي" if sdg in primary_sdgs else ""
                    
                    with sdg_cols[i % 3]:
                        st.markdown(f"""
                            <div class="sdg-badge {primary_class}">
                                الهدف {sdg}: {SDG_KEYWORDS[sdg]['name']}{primary_text}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #64748B; text-align: center; padding: 20px;">لم يتم العثور على أهداف مرتبطة بالمشروع</p>', unsafe_allow_html=True)
            
            # معلومات إضافية
            st.markdown('<div class="section-title">تفاصيل التحليل</div>', unsafe_allow_html=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; height: 100%;">
                        <div style="color: #64748B; margin-bottom: 5px;">توجه المشروع</div>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #0F172A;">{get_project_trend(metrics)}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; height: 100%;">
                        <div style="color: #64748B; margin-bottom: 5px;">عدد الأهداف</div>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #0F172A;">{metrics['sdg_count']} أهداف</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_info3:
                st.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; height: 100%;">
                        <div style="color: #64748B; margin-bottom: 5px;">درجة التوازن</div>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #0F172A;">{metrics['balance_score']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # توزيع الأهداف على الأبعاد
            st.markdown("""
                <div style="margin-top: 20px; background: white; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0;">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                        <div>
                            <div style="color: #64748B; margin-bottom: 5px;">اجتماعي</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0F172A;">{metrics['dimensions']['social']}</div>
                            <div style="color: #94A3B8; font-size: 0.9rem;">{metrics['social_ratio']:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #64748B; margin-bottom: 5px;">اقتصادي</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0F172A;">{metrics['dimensions']['economic']}</div>
                            <div style="color: #94A3B8; font-size: 0.9rem;">{metrics['economic_ratio']:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #64748B; margin-bottom: 5px;">بيئي</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #0F172A;">{metrics['dimensions']['environmental']}</div>
                            <div style="color: #94A3B8; font-size: 0.9rem;">{metrics['environmental_ratio']:.1f}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # الخلاصة
            st.markdown(f"""
                <div class="summary-box">
                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 15px; color: #0F172A;">الخلاصة</div>
                    <p style="color: #334155; line-height: 1.7; margin-bottom: 15px;">
                        مشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong>، يستهدف <strong>{metrics['sdg_count']}</strong> 
                        من أهداف التنمية المستدامة. تبلغ نسبة النجاح المتوقعة <strong>{success_prob*100:.1f}%</strong>.
                    </p>
                    <p style="color: #334155; margin-top: 10px; padding-top: 15px; border-top: 1px solid #E2E8F0;">
                        <strong>التوصية:</strong> {'المشروع واعد ويمكن المضي قدماً' if success_pred == 1 else 'يحتاج المشروع إلى إعادة هيكلة لتحسين فرص النجاح'}
                    </p>
                </div>
            """, unsafe_allow_html=True)

# --- 13. معلومات النماذج (شريط جانبي) ---
with st.sidebar:
    st.markdown("### معلومات النظام")
    
    files_status = check_model_files()
    available_files = [f for f, exists in files_status.items() if exists]
    
    if available_files:
        st.success(f"✓ الملفات المتاحة: {len(available_files)}")
    else:
        st.warning("⚠️ لا توجد نماذج مدربة")
    
    st.markdown("---")
    st.markdown("### ✨ ميزات التحليل الذكي")
    st.markdown("""
    • استخلاص أهداف متعددة من النص
    • تحديد الأهداف الأساسية والثانوية
    • قواعد ذكية للكلمات المركبة
    • تحليل دقيق للأبعاد الثلاثة
    """)

# --- 14. التذييل ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; padding: 20px;'>المنصة الذكية لتحليل المشاريع التنموية 2024</div>", unsafe_allow_html=True)
