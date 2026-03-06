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

# --- 4. قاموس الكلمات المفتاحية الموسع ---
SDG_KEYWORDS = {
    1: {
        'name': 'القضاء على الفقر', 
        'keywords': [
            'فقر', 'فقراء', 'فقير', 'تمكين اقتصادي', 'دخل', 'مساعدات', 'ضمان اجتماعي', 'تكافل',
            'الطبقات الفقيرة', 'الأسر المحتاجة', 'الدخل المحدود', 'الإغاثة', 'المساعدات النقدية',
            'التمويل الأصغر', 'مشاريع صغيرة', 'الأسر المنتجة', 'التمكين الاقتصادي', 'فرص عمل للفقراء',
            'تحسين سبل العيش', 'مكافحة الفقر', 'القضاء على الفقر', 'ذوو الدخل المنخفض', 'العمالة اليومية',
            'الفئات الهشة', 'الأرامل', 'الأيتام', 'المناطق النائية', 'الأحياء العشوائية', 'الإسكان الشعبي',
            'الزكاة', 'الصدقات', 'الأوقاف', 'الرعاية الاجتماعية', 'البطالة', 'العاطلين',
            'poverty', 'poor', 'low income', 'social safety', 'cash assistance', 'microfinance'
        ]
    },
    2: {
        'name': 'القضاء على الجوع', 
        'keywords': [
            'جوع', 'جياع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 'مزارعين', 'أراضي زراعية',
            'سوء تغذية', 'مجاعة', 'إنتاج غذائي', 'سلاسل توريد غذائية', 'مخزون استراتيجي', 'البنك الزراعي',
            'الصوامع', 'المطاحن', 'المخابز', 'المواشي', 'الثروة الحيوانية', 'الدواجن', 'الأسماك',
            'الاستزراع السمكي', 'الصيد', 'الري', 'المساحات الخضراء', 'الاكتفاء الذاتي', 'الأمن الغذائي',
            'جودة الغذاء', 'سلامة الغذاء', 'الغذاء الصحي', 'التغذية المدرسية', 'وجبات', 'برامج غذائية',
            'محاربة الجوع', 'القضاء على الجوع', 'تحسين التغذية', 'الغذاء المستدام', 'زراعة مستدامة',
            'زراعة عضوية', 'قمح', 'شعير', 'تمور', 'نخيل', 'مواشي',
            'hunger', 'food security', 'agriculture', 'farming', 'crops', 'food', 'nutrition'
        ]
    },
    3: {
        'name': 'الصحة الجيدة', 
        'keywords': [
            'صحة', 'مستشفى', 'مستشفيات', 'مركز صحي', 'مراكز صحية', 'رعاية صحية', 'أمراض', 'لقاحات',
            'أدوية', 'علاج', 'صحة عامة', 'صحة الأم والطفل', 'الرعاية الأولية', 'الطوارئ', 'الإسعاف',
            'العيادات', 'المستوصفات', 'المراكز الطبية', 'التأمين الطبي', 'التغطية الصحية',
            'الخدمات العلاجية', 'الخدمات الوقائية', 'التوعية الصحية', 'الأمراض المزمنة', 'السرطان',
            'السكري', 'الضغط', 'القلب', 'الأمراض المعدية', 'الأوبئة', 'جائحة', 'كورونا', 'كوفيد',
            'الصيدليات', 'المستلزمات الطبية', 'المعدات الطبية', 'المختبرات', 'الصحة النفسية',
            'العلاج النفسي', 'طب', 'أطباء', 'تمريض', 'ممرضين', 'صحة المرأة', 'صحة الطفل', 'تطعيمات',
            'health', 'healthcare', 'medical', 'hospital', 'clinic', 'doctor', 'treatment'
        ]
    },
    4: {
        'name': 'التعليم الجيد', 
        'keywords': [
            'تعليم', 'مدرسة', 'مدارس', 'جامعة', 'جامعات', 'طلاب', 'طالبات', 'معلمين', 'معلمات',
            'مناهج', 'تدريب', 'محو أمية', 'التعليم الأساسي', 'التعليم الثانوي', 'التعليم العالي',
            'رياض أطفال', 'حضانات', 'الفصول الدراسية', 'المباني المدرسية', 'المختبرات التعليمية',
            'المكتبات', 'الأنشطة الطلابية', 'المنح الدراسية', 'الابتعاث', 'التعليم الفني',
            'التعليم المهني', 'مراكز التدريب', 'تنمية المهارات', 'التدريب التقني', 'الحاسب الآلي',
            'اللغات', 'التربية الخاصة', 'ذوي الاحتياجات الخاصة', 'صعوبات التعلم', 'محو الأمية',
            'تعليم الكبار', 'التعليم عن بعد', 'التعليم الإلكتروني', 'المنصات التعليمية',
            'جودة التعليم', 'تطوير التعليم', 'تعليم الفتيات', 'تعليم البنات', 'معلم', 'معلمة',
            'education', 'school', 'university', 'student', 'teacher', 'training'
        ]
    },
    5: {
        'name': 'المساواة بين الجنسين', 
        'keywords': [
            'مساواة', 'نساء', 'نسائية', 'فتيات', 'تمكين المرأة', 'عنف ضد المرأة', 'حقوق المرأة',
            'المساواة بين الجنسين', 'المساواة الجندرية', 'المساواة في الفرص', 'التمييز ضد المرأة',
            'العنف الأسري', 'التحرش', 'الزواج المبكر', 'العنف الجنسي', 'المرأة الريفية',
            'المرأة العاملة', 'القيادة النسائية', 'ريادة الأعمال النسائية', 'مشاريع نسائية',
            'جمعيات نسائية', 'مراكز المرأة', 'حماية المرأة', 'دعم المرأة', 'تمكين الفتيات',
            'تعليم الفتيات', 'صحة المرأة', 'المشاركة السياسية للمرأة', 'التمثيل النسائي',
            'gender equality', 'women', 'girls', 'female empowerment', 'gender equity'
        ]
    },
    6: {
        'name': 'المياه النظيفة', 
        'keywords': [
            'مياه', 'ماء', 'صرف صحي', 'محطات تنقية', 'شرب', 'ري', 'سدود', 'آبار', 'مياه شرب نظيفة',
            'تحلية المياه', 'محطات تحلية', 'شبكات المياه', 'خزانات المياه', 'آبار ارتوازية',
            'الآبار الجوفية', 'السدود', 'الوديان', 'مياه الأمطار', 'تصريف السيول', 'معالجة المياه',
            'مياه الصرف', 'الصرف الزراعي', 'الصرف الصناعي', 'مياه الصرف الصحي', 'محطات المعالجة',
            'إعادة تدوير المياه', 'الري الحديث', 'الري بالتنقيط', 'الري المحوري',
            'ترشيد استهلاك المياه', 'حماية مصادر المياه', 'المياه الجوفية', 'الينابيع', 'الآبار',
            'العيون', 'الأفلاج', 'شبكات الري', 'خزان', 'تحلية',
            'water', 'clean water', 'sanitation', 'sewage', 'irrigation', 'dam'
        ]
    },
    7: {
        'name': 'الطاقة النظيفة', 
        'keywords': [
            'طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'شبكة كهرباء', 'محطات توليد',
            'الطاقة الشمسية', 'الألواح الشمسية', 'الخلايا الشمسية', 'محطات شمسية', 'الطاقة الريحية',
            'توربينات الرياح', 'مزارع الرياح', 'الطاقة النووية', 'الطاقة المائية',
            'السدود الكهرومائية', 'الطاقة الكهرومائية', 'الطاقة الحرارية', 'الطاقة الحيوية',
            'الوقود الحيوي', 'الكتلة الحيوية', 'الهيدروجين الأخضر', 'الأمونيا الخضراء',
            'توليد الكهرباء', 'نقل الكهرباء', 'شبكات النقل', 'كفاءة الطاقة', 'ترشيد الطاقة',
            'الطاقة المستدامة', 'الطاقة النظيفة', 'التحول الطاقوي', 'الحياد الكربوني',
            'طاقة الرياح', 'الطاقة الشمسية', 'خلية شمسية', 'لوح شمسي',
            'energy', 'electricity', 'solar', 'renewable', 'clean energy'
        ]
    },
    8: {
        'name': 'العمل اللائق', 
        'keywords': [
            'عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'بطالة', 'مهارات مهنية', 'العمل اللائق',
            'ظروف العمل', 'بيئة العمل', 'حقوق العمال', 'النقابات العمالية', 'العمال', 'الموظفين',
            'العمالة الوافدة', 'العمالة المنزلية', 'العمالة الموسمية', 'العمل الحر', 'العمل عن بعد',
            'المرونة الوظيفية', 'الأجور', 'الرواتب', 'الحد الأدنى للأجور', 'التأمينات الاجتماعية',
            'التقاعد', 'الضمان الاجتماعي', 'السلامة المهنية', 'الصحة المهنية', 'إصابات العمل',
            'حوادث العمل', 'التدريب المهني', 'التأهيل الوظيفي', 'التطوير المهني', 'المهارات الوظيفية',
            'الباحثين عن عمل', 'خريجين', 'توظيف الخريجين', 'مشاريع صغيرة', 'ريادة أعمال',
            'employment', 'jobs', 'work', 'labor', 'decent work', 'unemployment'
        ]
    },
    9: {
        'name': 'الصناعة والابتكار', 
        'keywords': [
            'صناعة', 'صناعي', 'ابتكار', 'بنية تحتية', 'طرق', 'جسور', 'مصانع', 'تكنولوجيا',
            'القطاع الصناعي', 'المدن الصناعية', 'المناطق الصناعية', 'المصانع', 'المنشآت الصناعية',
            'التصنيع', 'الإنتاج الصناعي', 'المواد الخام', 'المواد الأولية', 'السلع المصنعة',
            'الابتكار التقني', 'الابتكار التكنولوجي', 'البحث والتطوير', 'الاختراعات',
            'براءات الاختراع', 'الملكية الفكرية', 'التقنيات الحديثة', 'التقنيات الناشئة',
            'التحول الرقمي', 'الذكاء الاصطناعي', 'إنترنت الأشياء', 'الروبوتات', 'الأتمتة',
            'الصناعة 4.0', 'الثورة الصناعية الرابعة', 'الطرق السريعة', 'الموانئ', 'المطارات',
            'السكك الحديدية', 'المترو', 'القطارات', 'شبكات النقل', 'اللوجستيات', 'سلاسل الإمداد',
            'industry', 'innovation', 'infrastructure', 'technology', 'manufacturing'
        ]
    },
    10: {
        'name': 'الحد من عدم المساواة', 
        'keywords': [
            'مساواة', 'شمولية', 'فئات مهمشة', 'ذوي احتياجات خاصة', 'تمكين', 'عدم المساواة',
            'الفجوة الاجتماعية', 'الفجوة الاقتصادية', 'الفجوة الرقمية', 'المناطق المهمشة',
            'الأرياف', 'المناطق النائية', 'المناطق الحدودية', 'سكان البادية', 'البدو', 'الرحل',
            'اللاجئين', 'النازحين', 'المهاجرين', 'الأقليات', 'ذوو الإعاقة', 'المعاقين',
            'المكفوفين', 'الصم', 'ذوي الهمم', 'الاحتياجات الخاصة', 'كبار السن', 'المسنين',
            'المسنات', 'المتقاعدين', 'الأطفال', 'الشمول المالي', 'الشمول الاجتماعي',
            'الاندماج الاجتماعي', 'التكامل المجتمعي', 'العدالة الاجتماعية', 'تكافؤ الفرص',
            'المساواة في الحقوق', 'التمييز', 'العنصرية', 'الفقراء', 'المحتاجين', 'محدودي الدخل',
            'inequality', 'inclusion', 'marginalized', 'disabled', 'refugees', 'elderly'
        ]
    },
    11: {
        'name': 'مدن مستدامة', 
        'keywords': [
            'مدن', 'مدينة', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل عام', 'بنية تحتية حضرية',
            'المدن المستدامة', 'المجتمعات المستدامة', 'المدن الذكية', 'التخطيط العمراني',
            'التطوير الحضري', 'التجديد الحضري', 'تطوير المدن', 'المناطق الحضرية',
            'المراكز الحضرية', 'المدن الكبرى', 'المدن الجديدة', 'المجتمعات العمرانية',
            'الإسكان الميسر', 'الإسكان الاجتماعي', 'الإسكان التنموي', 'الإسكان الشعبي',
            'الأحياء السكنية', 'المرافق العامة', 'الخدمات البلدية', 'النظافة', 'الإنارة العامة',
            'الطرق الداخلية', 'الأرصفة', 'الحدائق العامة', 'المتنزهات', 'المساحات الخضراء',
            'المواصلات العامة', 'الحافلات', 'مترو الأنفاق', 'الترام', 'القطار الكهربائي',
            'إدارة المخلفات', 'تدوير النفايات', 'النفايات الصلبة', 'المخلفات', 'السكن',
            'sustainable cities', 'urban planning', 'housing', 'transport', 'public transport'
        ]
    },
    12: {
        'name': 'استهلاك مسؤول', 
        'keywords': [
            'استهلاك', 'إنتاج', 'استدامة', 'كفاءة موارد', 'إعادة تدوير', 'الاستهلاك المستدام',
            'الإنتاج المستدام', 'أنماط الاستهلاك', 'ترشيد الاستهلاك', 'الاستهلاك المسؤول',
            'الاستهلاك الواعي', 'الاستهلاك الأخضر', 'المنتجات المستدامة', 'المواد المستدامة',
            'المواد المعاد تدويرها', 'إعادة التدوير', 'تدوير المخلفات', 'إعادة الاستخدام',
            'تقليل الاستهلاك', 'تقليل الهدر', 'هدر الطعام', 'الفاقد الغذائي', 'الاقتصاد الدائري',
            'التصميم المستدام', 'المنتجات الصديقة للبيئة', 'البصمة البيئية', 'البصمة الكربونية',
            'استدامة الموارد', 'كفاءة الموارد', 'كفاءة الطاقة', 'كفاءة المياه', 'ترشيد الطاقة',
            'ترشيد المياه', 'تدوير', 'فرز', 'مخلفات', 'نفايات', 'بلاستيك', 'ورق', 'زجاج', 'معادن',
            'responsible consumption', 'recycling', 'circular economy', 'sustainability'
        ]
    },
    13: {
        'name': 'العمل المناخي', 
        'keywords': [
            'مناخ', 'تغير مناخي', 'انبعاثات', 'احتباس حراري', 'كربون', 'التغير المناخي',
            'تغيرات المناخ', 'الاحتباس الحراري', 'الاحترار العالمي', 'غازات الاحتباس الحراري',
            'غازات الدفيئة', 'انبعاثات الكربون', 'الكربون', 'ثاني أكسيد الكربون', 'الميثان',
            'الانبعاثات الكربونية', 'الحياد الكربوني', 'صفر كربون', 'خفض الانبعاثات',
            'التخفيف من التغير المناخي', 'التكيف مع التغير المناخي', 'مقاومة المناخ',
            'الطقس المتطرف', 'الظواهر الجوية', 'الكوارث الطبيعية', 'الفيضانات', 'الجفاف',
            'العواصف', 'الأعاصير', 'حرائق الغابات', 'ارتفاع منسوب البحر', 'ذوبان الجليد',
            'الطاقة النظيفة', 'الطاقة المتجددة', 'الحلول المناخية', 'الاستدامة البيئية',
            'climate', 'climate change', 'emissions', 'carbon', 'global warming'
        ]
    },
    14: {
        'name': 'الحياة تحت الماء', 
        'keywords': [
            'بحار', 'بحر', 'محيطات', 'محيط', 'أسماك', 'سمك', 'سواحل', 'ساحلي', 'ثروة بحرية',
            'صيد', 'الحياة البحرية', 'الكائنات البحرية', 'النظم البيئية البحرية', 'الشعاب المرجانية',
            'المرجان', 'السلاحف البحرية', 'الثدييات البحرية', 'الحيتان', 'الدلافين', 'الأسماك',
            'المخزون السمكي', 'الثروة السمكية', 'مصائد الأسماك', 'الصيد الجائر', 'الاستزراع السمكي',
            'المزارع السمكية', 'الأحياء المائية', 'الربيان', 'الجمبري', 'تلوث البحار',
            'التلوث البحري', 'المخلفات البحرية', 'البلاستيك في المحيطات', 'المحميات البحرية',
            'المناطق البحرية المحمية', 'حماية السواحل', 'إدارة السواحل', 'الاقتصاد الأزرق',
            'الموارد البحرية', 'الموارد الساحلية', 'السياحة البحرية',
            'oceans', 'sea', 'marine', 'fisheries', 'coral reefs', 'marine conservation'
        ]
    },
    15: {
        'name': 'الحياة في البر', 
        'keywords': [
            'بيئة', 'غابات', 'غابة', 'تنوع أحيائي', 'تنوع بيولوجي', 'محيات طبيعية', 'حيوانات',
            'نباتات', 'النظم البيئية الأرضية', 'النظم البيئية البرية', 'الغابات', 'الأحراش',
            'البراري', 'المحميات الطبيعية', 'المحميات البرية', 'المناطق المحمية', 'الحدائق الوطنية',
            'الحياة البرية', 'الحيوانات البرية', 'الطيور', 'الكائنات الفطرية',
            'الكائنات المهددة بالانقراض', 'الأنواع المهددة', 'الأنواع النادرة', 'التنوع الحيوي',
            'التنوع البيولوجي', 'الموارد الوراثية', 'النباتات', 'الأشجار', 'النباتات الطبيعية',
            'الغطاء النباتي', 'التصحر', 'مكافحة التصحر', 'تدهور الأراضي', 'تآكل التربة',
            'إعادة التشجير', 'زراعة الغابات', 'التشجير', 'الاستدامة البيئية',
            'environment', 'forests', 'biodiversity', 'wildlife', 'conservation'
        ]
    },
    16: {
        'name': 'السلام والعدالة', 
        'keywords': [
            'سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'سيادة القانون', 'الأمن والسلام',
            'الاستقرار', 'الأمن المجتمعي', 'الأمن الوطني', 'مكافحة الإرهاب', 'التطرف',
            'مكافحة الجريمة', 'الجرائم', 'النزاعات', 'حل النزاعات', 'الوساطة', 'المصالحة',
            'المصالحة الوطنية', 'بناء السلام', 'حفظ السلام', 'بعثات السلام', 'العدالة الجنائية',
            'العدالة الناجزة', 'العدالة الانتقالية', 'القضاء', 'المحاكم', 'النيابة العامة',
            'التحكيم', 'الحوكمة الرشيدة', 'الإدارة الرشيدة', 'مكافحة الفساد', 'الشفافية',
            'المساءلة', 'سيادة القانون', 'حكم القانون', 'المؤسسات الحكومية', 'الإصلاح المؤسسي',
            'حقوق الإنسان', 'الحريات العامة', 'الحقوق المدنية', 'الحقوق السياسية',
            'peace', 'justice', 'governance', 'rule of law', 'anti-corruption'
        ]
    },
    17: {
        'name': 'الشراكات', 
        'keywords': [
            'شراكات', 'شراكة', 'تعاون دولي', 'تمويل', 'منح', 'قروض', 'مساعدات دولية',
            'الشراكات الدولية', 'الشراكات الإقليمية', 'الشراكات المحلية', 'التعاون المشترك',
            'التعاون الثنائي', 'التعاون متعدد الأطراف', 'المنظمات الدولية', 'الأمم المتحدة',
            'البنك الدولي', 'صندوق النقد الدولي', 'المانحين', 'الجهات المانحة', 'الدول المانحة',
            'المساعدات الإنمائية', 'المساعدات الإنسانية', 'المساعدات الفنية', 'الدعم الفني',
            'التمويل المشترك', 'التمويل الدولي', 'الاستثمار الأجنبي', 'الاستثمار المباشر',
            'القطاع الخاص', 'الاستثمار الخاص', 'المنظمات غير الحكومية', 'المجتمع المدني',
            'المؤسسات الخيرية', 'الأوقاف', 'الصناديق التنموية', 'الصناديق السيادية',
            'partnerships', 'cooperation', 'funding', 'grants', 'aid'
        ]
    }
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
    {'triggers': ['تعليم الفتيات', 'تمكين الفتيات'], 'target_sdgs': [4, 5], 'primary': 4},
    {'triggers': ['مياه نظيفة للمجتمعات الريفية', 'مياه ريفية'], 'target_sdgs': [6, 10], 'primary': 6},
    {'triggers': ['طاقة شمسية للمجتمعات النائية', 'كهرباء قرى'], 'target_sdgs': [7, 10], 'primary': 7},
    {'triggers': ['صحة المرأة', 'صحة الأم'], 'target_sdgs': [3, 5], 'primary': 3},
    {'triggers': ['طفل', 'أطفال', 'الطفولة'], 'target_sdgs': [3, 4], 'primary': 4},
    {'triggers': ['شباب', 'تمكين الشباب'], 'target_sdgs': [4, 8], 'primary': 8},
    {'triggers': ['ذوي إعاقة', 'ذوي الاحتياجات الخاصة'], 'target_sdgs': [3, 10], 'primary': 10},
    {'triggers': ['لاجئين', 'نازحين'], 'target_sdgs': [1, 10, 16], 'primary': 10},
    {'triggers': ['سياحة مستدامة', 'سياحة بيئية'], 'target_sdgs': [8, 12, 14], 'primary': 12},
    {'triggers': ['اقتصاد دائري', 'تدوير'], 'target_sdgs': [9, 12], 'primary': 12},
    {'triggers': ['مدن ذكية', 'مدن مستدامة'], 'target_sdgs': [9, 11], 'primary': 11},
    {'triggers': ['حماية البيئة', 'استدامة بيئية'], 'target_sdgs': [13, 14, 15], 'primary': 13},
    {'triggers': ['مشاريع صغيرة', 'ريادة أعمال'], 'target_sdgs': [8, 9], 'primary': 8},
    {'triggers': ['تنمية مجتمعية', 'تنمية محلية'], 'target_sdgs': [1, 11], 'primary': 11}
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
    
    # البحث عن القواعد الخاصة (الكلمات المركبة)
    for rule in MULTI_SDG_RULES:
        for trigger in rule['triggers']:
            if trigger in text:
                detected_sdgs.extend(rule['target_sdgs'])
                matched_keywords.append(trigger)
                if rule['primary'] not in primary_sdgs:
                    primary_sdgs.append(rule['primary'])
                break
    
    # البحث في الكلمات المفتاحية العادية
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

# --- 8. تحسين نسبة النجاح بإضافة الميزانية والمستفيدين ---
def enhance_success_with_budget(original_prob, p_budget, p_ben):
    """تحسين نسبة النجاح بناءً على كفاءة الميزانية والعائد الاجتماعي"""
    
    if p_budget <= 0 or p_ben <= 0:
        return original_prob, 0, 0, 0
    
    # حساب التكلفة لكل مستفيد
    cost_per_person = p_budget / p_ben
    
    # حساب العائد الاجتماعي
    sroi = (p_ben * original_prob) / (p_budget / 1000)
    
    # معامل تحسين الميزانية
    budget_factor = 0
    
    # مكافأة الكفاءة (التكلفة المنخفضة لكل مستفيد)
    if cost_per_person < 5000:
        budget_factor += 0.10  # +10% كفاءة ممتازة
    elif cost_per_person < 15000:
        budget_factor += 0.05   # +5% كفاءة جيدة
    elif cost_per_person > 100000:
        budget_factor -= 0.10   # -10% تكلفة عالية جداً
    elif cost_per_person > 50000:
        budget_factor -= 0.05   # -5% تكلفة مرتفعة
    
    # مكافأة العائد الاجتماعي
    if sroi > 15:
        budget_factor += 0.08   # +8% عائد استثنائي
    elif sroi > 8:
        budget_factor += 0.03   # +3% عائد جيد
    elif sroi < 0.5:
        budget_factor -= 0.08   # -8% عائد ضعيف
    elif sroi < 1:
        budget_factor -= 0.03   # -3% عائد منخفض
    
    # مكافأة المشاريع الكبيرة (أكثر من 10,000 مستفيد)
    if p_ben > 10000:
        budget_factor += 0.05   # +5% تأثير مجتمعي واسع
    
    # تطبيق التحسين
    enhanced_prob = original_prob + budget_factor
    
    # التأكد من أن النسبة بين 0 و 1
    enhanced_prob = max(0, min(enhanced_prob, 1.0))
    
    return enhanced_prob, budget_factor, cost_per_person, sroi

# --- 9. نظام التوصيات المنطقية (مرتبطة بخصائص المشروع) ---
def generate_logical_recommendations(metrics, p_cat, p_budget, p_ben, success_prob, cost_per_person, sroi, detected_sdgs, primary_sdgs):
    """توليد توصيات منطقية مرتبطة بخصائص المشروع"""
    
    recommendations = []
    weaknesses = []
    strengths = []
    
    # ============================================
    # 1. نقاط القوة (بناءً على التحليل)
    # ============================================
    
    # قوة عدد الأهداف
    if metrics['sdg_count'] >= 4:
        strengths.append(f"✅ المشروع يغطي {metrics['sdg_count']} أهداف تنموية متنوعة")
    elif metrics['sdg_count'] >= 3:
        strengths.append(f"✅ المشروع يغطي {metrics['sdg_count']} أهداف تنموية")
    
    # قوة التوازن
    if metrics['balance_score'] > 80:
        strengths.append(f"✅ توازن ممتاز بين الأبعاد التنموية ({metrics['balance_score']:.1f}%)")
    elif metrics['balance_score'] > 70:
        strengths.append(f"✅ توازن جيد بين الأبعاد التنموية ({metrics['balance_score']:.1f}%)")
    
    # قوة كفاءة الميزانية
    if cost_per_person < 5000:
        strengths.append(f"✅ كفاءة تشغيلية ممتازة ({cost_per_person:,.0f} ريال لكل مستفيد)")
    elif cost_per_person < 15000:
        strengths.append(f"✅ كفاءة تشغيلية جيدة ({cost_per_person:,.0f} ريال لكل مستفيد)")
    
    # قوة العائد الاجتماعي
    if sroi > 15:
        strengths.append(f"✅ عائد اجتماعي استثنائي ({sroi:.1f}x)")
    elif sroi > 8:
        strengths.append(f"✅ عائد اجتماعي مرتفع ({sroi:.1f}x)")
    
    # قوة الأهداف الأساسية
    if len(primary_sdgs) > 0:
        primary_names = [SDG_KEYWORDS[sdg]['name'] for sdg in primary_sdgs]
        strengths.append(f"✅ تركيز واضح على: {', '.join(primary_names)}")
    
    # ============================================
    # 2. نقاط الضعف (بناءً على التحليل)
    # ============================================
    
    # ضعف عدد الأهداف
    if metrics['sdg_count'] == 0:
        weaknesses.append("⚠️ المشروع غير مرتبط بأهداف التنمية المستدامة")
    elif metrics['sdg_count'] == 1:
        weaknesses.append("⚠️ المشروع يركز على هدف تنموي واحد فقط")
    elif metrics['sdg_count'] == 2:
        weaknesses.append("⚠️ عدد الأهداف التنموية محدود (هدفان)")
    
    # ضعف التوازن
    if metrics['balance_score'] < 30:
        weaknesses.append(f"⚠️ اختلال كبير في التوازن بين الأبعاد التنموية ({metrics['balance_score']:.1f}%)")
    elif metrics['balance_score'] < 50:
        weaknesses.append(f"⚠️ توازن ضعيف بين الأبعاد التنموية ({metrics['balance_score']:.1f}%)")
    
    # ضعف كفاءة الميزانية
    if cost_per_person > 100000:
        weaknesses.append(f"⚠️ تكلفة مرتفعة جداً لكل مستفيد ({cost_per_person:,.0f} ريال)")
    elif cost_per_person > 50000:
        weaknesses.append(f"⚠️ تكلفة مرتفعة لكل مستفيد ({cost_per_person:,.0f} ريال)")
    
    # ضعف العائد الاجتماعي
    if sroi < 0.5:
        weaknesses.append(f"⚠️ عائد اجتماعي منخفض جداً ({sroi:.1f}x)")
    elif sroi < 1:
        weaknesses.append(f"⚠️ عائد اجتماعي منخفض ({sroi:.1f}x)")
    
    # ضعف عدد المستفيدين مقارنة بالميزانية
    if p_ben < 100 and p_budget > 500000:
        weaknesses.append("⚠️ عدد المستفيدين قليل مقارنة بحجم الميزانية")
    
    # ============================================
    # 3. التوصيات المنطقية (مرتبطة بخصائص المشروع)
    # ============================================
    
    # توصيات حسب عدد الأهداف
    if metrics['sdg_count'] == 0:
        recommendations.append("🎯 **تحديد الأهداف**: المشروع بحاجة للربط بأهداف التنمية المستدامة. حدد 2-3 أهداف رئيسية للمشروع")
    elif metrics['sdg_count'] == 1:
        # معرفة الهدف الحالي
        current_sdg = detected_sdgs[0]
        current_name = SDG_KEYWORDS[current_sdg]['name']
        
        # اقتراح هدف إضافي مناسب
        if current_sdg in SDG_DIMENSIONS['social']:
            recommendations.append(f"🎯 **تنويع الأهداف**: المشروع يركز على '{current_name}'. أضف هدفاً اقتصادياً أو بيئياً لتعزيز التكامل")
        elif current_sdg in SDG_DIMENSIONS['economic']:
            recommendations.append(f"🎯 **تنويع الأهداف**: المشروع يركز على '{current_name}'. أضف هدفاً اجتماعياً مثل 'الحد من عدم المساواة'")
        elif current_sdg in SDG_DIMENSIONS['environmental']:
            recommendations.append(f"🎯 **تنويع الأهداف**: المشروع يركز على '{current_name}'. أضف هدفاً اجتماعياً لتعزيز الأثر المجتمعي")
    
    # توصيات حسب التوازن
    if metrics['balance_score'] < 40:
        # تحديد البعد المهمل
        dims = {
            'الاجتماعي': metrics['social_ratio'],
            'الاقتصادي': metrics['economic_ratio'],
            'البيئي': metrics['environmental_ratio']
        }
        min_dim = min(dims, key=dims.get)
        recommendations.append(f"⚖️ **تحسين التوازن**: التركيز الحالي على الأبعاد الأخرى. أضف أهدافاً من البعد {min_dim} لتحقيق توازن أفضل")
    
    # توصيات حسب الميزانية والتكلفة
    if cost_per_person > 100000:
        recommendations.append("💰 **إعادة هيكلة الميزانية**: التكلفة لكل مستفيد مرتفعة جداً. ابحث عن طرق لخفض التكاليف أو زيادة عدد المستفيدين")
    elif cost_per_person > 50000:
        recommendations.append("💰 **ترشيد الإنفاق**: خفض التكاليف بنسبة 15-20% مع الحفاظ على جودة المشروع")
    
    if p_ben < 100 and p_budget > 500000:
        recommendations.append("📈 **توسيع النطاق**: الميزانية كبيرة مقارنة بعدد المستفيدين. ابحث عن فرص لتوسيع المشروع ليشمل مناطق أو فئات إضافية")
    
    # توصيات حسب العائد الاجتماعي
    if sroi < 1:
        recommendations.append("📊 **تحسين الأثر**: العائد على الاستثمار منخفض. ركز على الفئات الأكثر احتياجاً واستهدف المجتمعات المهمشة لزيادة الأثر")
    
    # توصيات حسب القطاع
    if p_cat == "تعليمي":
        if metrics['sdg_count'] < 2:
            recommendations.append("📚 **التكامل التعليمي**: المشاريع التعليمية الناجحة تدمج بين التعليم وأهداف أخرى مثل 'الحد من عدم المساواة' أو 'العمل اللائق'")
        if metrics['balance_score'] < 50:
            recommendations.append("📚 **تنويع الأبعاد**: أضف أنشطة تركز على الجانب الاقتصادي أو البيئي للمشروع التعليمي")
    
    elif p_cat == "صحي":
        if metrics['sdg_count'] < 2:
            recommendations.append("🏥 **التكامل الصحي**: المشاريع الصحية الأكثر نجاحاً تدمج بين الرعاية الصحية وأهداف اجتماعية مثل 'الحد من عدم المساواة'")
        if metrics['dimensions']['social'] == 0:
            recommendations.append("🏥 **البعد الاجتماعي**: أضف أهدافاً اجتماعية للمشروع الصحي مثل استهداف الفئات المهمشة")
    
    elif p_cat == "بيئي":
        if metrics['sdg_count'] < 2:
            recommendations.append("🌍 **التكامل البيئي**: المشاريع البيئية الناجحة تدمج بين حماية البيئة وأهداف اقتصادية أو اجتماعية")
        if metrics['dimensions']['social'] == 0:
            recommendations.append("🌍 **البعد الاجتماعي**: أضف أهدافاً اجتماعية للمشروع البيئي مثل توفير فرص عمل خضراء")
    
    elif p_cat == "اقتصادي":
        if metrics['dimensions']['social'] == 0:
            recommendations.append("📈 **البعد الاجتماعي**: المشاريع الاقتصادية الأكثر استدامة تدمج أهدافاً اجتماعية مثل العمل اللائق والحد من البطالة")
        if metrics['dimensions']['environmental'] == 0:
            recommendations.append("📈 **البعد البيئي**: أضف أهدافاً بيئية للمشروع الاقتصادي مثل كفاءة الطاقة أو الاستهلاك المسؤول")
    
    elif p_cat == "اجتماعي":
        if metrics['dimensions']['economic'] == 0:
            recommendations.append("👥 **البعد الاقتصادي**: المشاريع الاجتماعية الناجحة تدمج أهدافاً اقتصادية مثل تمكين الأسر المنتجة وخلق فرص عمل")
    
    # توصيات استثمارية حسب نسبة النجاح
    if success_prob < 0.4:
        recommendations.append("⚠️ **موقف استثماري**: نسبة المخاطرة عالية. يوصى بإعادة دراسة المشروع أو البحث عن مصادر تمويل غير تقليدية")
    elif success_prob < 0.6:
        recommendations.append("📋 **موقف استثماري**: مخاطرة متوسطة. يوصى بتمويل مرحلي مع متابعة دقيقة للمؤشرات")
    elif success_prob > 0.8:
        recommendations.append("✨ **موقف استثماري**: فرصة استثمارية واعدة. يوصى بالتمويل والتوسع في المشروع")
    
    # توصيات للمشاريع الكبيرة
    if p_budget > 1000000 and success_prob < 0.6:
        recommendations.append("🔍 **تقييم المخاطر**: الميزانية الكبيرة تتطلب دراسة جدوى معمقة. يوصى بتدقيق إضافي قبل الاعتماد")
    
    # توصيات للأهداف الأساسية
    if len(primary_sdgs) > 0:
        for sdg in primary_sdgs:
            if sdg == 10:  # الحد من عدم المساواة
                if metrics['dimensions']['social'] > 1:
                    recommendations.append("🤝 **تعزيز الشمولية**: المشروع يستهدف الفئات المهمشة. يوصى بتوسيع نطاق الشراكات مع مؤسسات المجتمع المدني")
            elif sdg == 4:  # التعليم
                if metrics['balance_score'] < 50:
                    recommendations.append("📖 **تطوير التعليم**: أضف مكونات تدريب مهني لربط التعليم بسوق العمل")
            elif sdg == 3:  # الصحة
                if p_ben < 500:
                    recommendations.append("💊 **توسيع التغطية الصحية**: المشروع الصحي يمكن أن يصل لعدد أكبر من المستفيدين عبر الشراكات مع المراكز الصحية")
    
    # إزالة التكرارات
    recommendations = list(dict.fromkeys(recommendations))
    
    # تحديد مستوى الثقة
    confidence_level = "منخفضة"
    if success_prob > 0.7 and metrics['balance_score'] > 60 and p_ben > 100:
        confidence_level = "عالية جداً"
    elif success_prob > 0.5:
        confidence_level = "متوسطة"
    
    return strengths, weaknesses, recommendations[:6], confidence_level

# --- 10. تحميل النماذج ---
models = load_models_safe()

# --- 11. التصميم ---
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
        font-size: 1rem;
    }
    
    .weakness-item {
        color: #991B1B;
        padding: 8px 0;
        border-bottom: 1px solid #FECACA;
        font-size: 1rem;
    }
    
    .recommendation-item {
        color: #1E3A8A;
        padding: 12px 0;
        border-bottom: 1px solid #BFDBFE;
        font-size: 1rem;
        line-height: 1.6;
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
    
    .budget-impact-box {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0 20px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .footer {
        text-align: center;
        color: #9CA3AF;
        padding: 30px 0 20px 0;
        border-top: 1px solid #E5E7EB;
        margin-top: 50px;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 12. العنوان ---
st.markdown("<h1>المنصة الذكية لتحليل المشاريع التنموية</h1>", unsafe_allow_html=True)

# --- 13. نموذج الإدخال ---
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

# --- 14. التحليل والنتائج مع التوصيات المنطقية ---
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
                    
                    original_prob = (weight_ann * ann_prob + weight_xgb * xgb_prob)
                    
                except Exception as e:
                    original_prob = predict_success_fallback(metrics)
            else:
                original_prob = predict_success_fallback(metrics)
            
            # تحسين نسبة النجاح بإضافة الميزانية والمستفيدين
            enhanced_prob, budget_factor, cost_per_person, sroi = enhance_success_with_budget(
                original_prob, p_budget, p_ben
            )
            
            # التصنيف النهائي
            threshold = 0.4
            success_pred = 1 if enhanced_prob >= threshold else 0
            
            # --- توليد التوصيات المنطقية ---
            strengths, weaknesses, recommendations, confidence_level = generate_logical_recommendations(
                metrics, p_cat, p_budget, p_ben, enhanced_prob, cost_per_person, sroi, detected_sdgs, primary_sdgs
            )
            
            # عرض نسبة النجاح
            status_color = "#10B981" if success_pred == 1 else "#EF4444"
            status_text = "ناجح" if success_pred == 1 else "غير ناجح"
            
            st.markdown(f"""
                <div class="success-card">
                    <div class="label">نسبة نجاح المشروع</div>
                    <div class="value">{enhanced_prob*100:.1f}%</div>
                    <div class="status" style="background: {status_color}15; color: {status_color};">
                        {status_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # عرض تأثير الميزانية
            if budget_factor != 0:
                impact_color = "#10B981" if budget_factor > 0 else "#EF4444"
                impact_symbol = "▲" if budget_factor > 0 else "▼"
                
                st.markdown(f"""
                    <div class="budget-impact-box">
                        <div>
                            <span style="color: #6B7280;">تأثير كفاءة الميزانية:</span>
                            <span style="color: {impact_color}; font-weight: 600; margin-right: 10px;">
                                {impact_symbol} {abs(budget_factor*100):.1f}%
                            </span>
                        </div>
                        <div>
                            <span style="color: #6B7280;">التكلفة لكل مستفيد:</span>
                            <span style="color: #0F172A; font-weight: 600; margin-right: 5px;">
                                {cost_per_person:,.0f} ريال
                            </span>
                        </div>
                        <div>
                            <span style="color: #6B7280;">العائد الاجتماعي:</span>
                            <span style="color: #0F172A; font-weight: 600; margin-right: 5px;">
                                {sroi:.1f}x
                            </span>
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
            
            # عرض نقاط القوة
            if strengths:
                st.markdown("""
                    <div class="strengths-box">
                        <h4 style="color: #065F46; margin-bottom: 15px;">✅ نقاط القوة</h4>
                """, unsafe_allow_html=True)
                
                for strength in strengths:
                    st.markdown(f'<div class="strength-item">✓ {strength}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # عرض نقاط الضعف
            if weaknesses:
                st.markdown("""
                    <div class="weaknesses-box">
                        <h4 style="color: #991B1B; margin-bottom: 15px;">⚠️ نقاط تحتاج تحسين</h4>
                """, unsafe_allow_html=True)
                
                for weakness in weaknesses:
                    st.markdown(f'<div class="weakness-item">• {weakness}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # عرض التوصيات
            if recommendations:
                st.markdown("""
                    <div class="recommendations-box">
                        <h4 style="color: #1E3A8A; margin-bottom: 20px;">💡 توصيات للتحسين</h4>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
                
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
            else:
                st.info("لم يتم العثور على أهداف مرتبطة بالمشروع")
            
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
            
            # الخلاصة مع توصية للمستثمر
            investment_advice = ""
            if confidence_level == "عالية جداً" and enhanced_prob > 0.7:
                investment_advice = "✅ فرصة استثمارية واعدة - يوصى بالتمويل"
            elif confidence_level == "متوسطة" and enhanced_prob > 0.5:
                investment_advice = "⚠️ استثمار متوسط المخاطرة - يوصى بتمويل مشروط بمتابعة"
            else:
                investment_advice = "❌ مخاطرة عالية - يوصى بإعادة الدراسة قبل التمويل"
            
            st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-title">📋 خلاصة التحليل وقرار المستثمر</div>
                    <div class="summary-text">
                        <p>مشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong>، 
                        بميزانية <strong>{p_budget:,.0f} ريال</strong> لـ <strong>{p_ben:,} مستفيد</strong>.</p>
                        <p>نسبة النجاح المتوقعة <strong>{enhanced_prob*100:.1f}%</strong>، 
                        ومستوى الثقة للمستثمر <strong>{confidence_level}</strong>.</p>
                        <div style="margin-top: 15px; padding: 15px; background: #F3F4F6; border-radius: 8px;">
                            <strong>توصية استثمارية:</strong> {investment_advice}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 15. التذييل ---
st.markdown('<div class="footer">المنصة الذكية لتحليل المشاريع التنموية 2026</div>', unsafe_allow_html=True)
