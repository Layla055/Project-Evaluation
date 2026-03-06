import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import os
import re
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# --- 1. إعدادات الهوية المؤسسية ---
st.set_page_config(page_title="المنصة الذكية لتحليل المشاريع التنموية", layout="wide", initial_sidebar_state="collapsed")

# --- 2. تحميل النماذج المدربة ---
@st.cache_resource
def load_trained_models():
    """تحميل النماذج المدربة من ملفات"""
    try:
        ann_model = load_model('hybrid_ann.h5')
        xgb_model = joblib.load('hybrid_xgb.pkl')
        scaler = joblib.load('scaler.pkl')
        config = joblib.load('config.pkl')
        
        return {
            'ann': ann_model,
            'xgb': xgb_model,
            'scaler': scaler,
            'config': config
        }
    except Exception as e:
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        return None

# --- 3. قاموس الكلمات المفتاحية (موسع جداً) ---
SDG_KEYWORDS = {
    1: {
        'name': 'القضاء على الفقر',
        'keywords': [
            'فقر', 'فقراء', 'تمكين اقتصادي', 'دخل', 'مساعدات', 'ضمان اجتماعي', 'تكافل', 'مسكن', 'غذاء',
            'بطالة', 'عمالة غير منتظمة', 'حد الفقر', 'الطبقات الفقيرة', 'الأسر المحتاجة', 'الدخل المحدود',
            'الإغاثة', 'المساعدات النقدية', 'التمويل الأصغر', 'مشاريع صغيرة', 'التأمين الصحي', 'الرعاية الاجتماعية',
            'الزكاة', 'الصدقات', 'الأوقاف', 'الإسكان الشعبي', 'الأحياء العشوائية', 'المناطق النائية',
            'الفئات الهشة', 'الأرامل', 'الأيتام', 'ذوو الدخل المنخفض', 'العمالة اليومية', 'الأسر المنتجة',
            'التمكين الاقتصادي', 'فرص عمل للفقراء', 'تحسين سبل العيش', 'مكافحة الفقر', 'القضاء على الفقر',
            'poverty', 'poor', 'low income', 'social safety', 'cash assistance', 'microfinance'
        ]
    },
    2: {
        'name': 'القضاء على الجوع',
        'keywords': [
            'جوع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 'مزارعين', 'أراضي زراعية',
            'سوء تغذية', 'مجاعة', 'محاصيل زراعية', 'إنتاج غذائي', 'سلاسل توريد غذائية', 'مخزون استراتيجي',
            'البنك الزراعي', 'الصوامع', 'المطاحن', 'المخابز', 'المواشي', 'الثروة الحيوانية', 'الدواجن',
            'الأسماك', 'الاستزراع السمكي', 'الصيد', 'الري', 'المساحات الخضراء', 'الاكتفاء الذاتي',
            'الأمن الغذائي', 'جودة الغذاء', 'سلامة الغذاء', 'الغذاء الصحي', 'التغذية المدرسية', 'وجبات',
            'برامج غذائية', 'محاربة الجوع', 'القضاء على الجوع', 'تحسين التغذية', 'الغذاء المستدام',
            'hunger', 'food security', 'agriculture', 'nutrition', 'farming', 'crops', 'food supply'
        ]
    },
    3: {
        'name': 'الصحة الجيدة',
        'keywords': [
            'صحة', 'مستشفى', 'مركز صحي', 'رعاية صحية', 'أمراض', 'لقاحات', 'أدوية', 'علاج',
            'صحة عامة', 'صحة الأم والطفل', 'الرعاية الأولية', 'الطوارئ', 'الإسعاف', 'العيادات',
            'المستوصفات', 'المراكز الطبية', 'التأمين الطبي', 'التغطية الصحية', 'الخدمات العلاجية',
            'الخدمات الوقائية', 'التوعية الصحية', 'الأمراض المزمنة', 'السرطان', 'السكري', 'الضغط',
            'القلب', 'الأمراض المعدية', 'الأوبئة', 'جائحة', 'كورونا', 'كوفيد', 'فيروسات',
            'الصيدليات', 'المستلزمات الطبية', 'المعدات الطبية', 'الأجهزة الطبية', 'الأشعة', 'المختبرات',
            'التحاليل', 'الفحوصات', 'الاستشارات الطبية', 'الصحة النفسية', 'العلاج النفسي',
            'health', 'hospital', 'clinic', 'medical', 'healthcare', 'treatment', 'vaccine', 'medicine'
        ]
    },
    4: {
        'name': 'التعليم الجيد',
        'keywords': [
            'تعليم', 'مدرسة', 'جامعة', 'طلاب', 'معلمين', 'مناهج', 'تدريب', 'محو أمية',
            'التعليم الأساسي', 'التعليم الثانوي', 'التعليم العالي', 'رياض أطفال', 'حضانات',
            'الفصول الدراسية', 'المباني المدرسية', 'المختبرات التعليمية', 'المكتبات', 'الأنشطة الطلابية',
            'المنح الدراسية', 'الابتعاث', 'التعليم الفني', 'التعليم المهني', 'مراكز التدريب',
            'تنمية المهارات', 'التدريب التقني', 'الحاسب الآلي', 'اللغات', 'التربية الخاصة',
            'ذوي الاحتياجات الخاصة', 'صعوبات التعلم', 'محو الأمية', 'تعليم الكبار', 'التعليم عن بعد',
            'التعليم الإلكتروني', 'المنصات التعليمية', 'المحتوى التعليمي', 'المناهج المطورة',
            'جودة التعليم', 'تطوير التعليم', 'تحسين المخرجات التعليمية', 'التميز التعليمي',
            'education', 'school', 'university', 'students', 'teachers', 'training', 'learning'
        ]
    },
    5: {
        'name': 'المساواة بين الجنسين',
        'keywords': [
            'مساواة', 'نساء', 'فتيات', 'تمكين المرأة', 'عنف ضد المرأة', 'حقوق المرأة',
            'المساواة بين الجنسين', 'المساواة الجندرية', 'المساواة في الفرص', 'التمييز ضد المرأة',
            'العنف الأسري', 'التحرش', 'الزواج المبكر', 'ختان الإناث', 'العنف الجنسي',
            'المرأة الريفية', 'المرأة العاملة', 'القيادة النسائية', 'ريادة الأعمال النسائية',
            'مشاريع نسائية', 'جمعيات نسائية', 'مراكز المرأة', 'حماية المرأة', 'دعم المرأة',
            'تمكين الفتيات', 'تعليم الفتيات', 'صحة المرأة', 'الحقوق الإنجابية', 'المشاركة السياسية للمرأة',
            'التمثيل النسائي', 'المرأة في المناصب القيادية', 'التوازن بين الجنسين',
            'gender equality', 'women', 'girls', 'female empowerment', 'gender equity'
        ]
    },
    6: {
        'name': 'المياه النظيفة',
        'keywords': [
            'مياه', 'صرف صحي', 'محطات تنقية', 'شرب', 'ري', 'سدود', 'آبار',
            'مياه شرب نظيفة', 'تحلية المياه', 'محطات تحلية', 'شبكات المياه', 'خزانات المياه',
            'آبار ارتوازية', 'الآبار الجوفية', 'السدود', 'الوديان', 'مياه الأمطار', 'تصريف السيول',
            'معالجة المياه', 'مياه الصرف', 'الصرف الزراعي', 'الصرف الصناعي', 'مياه الصرف الصحي',
            'محطات المعالجة', 'إعادة تدوير المياه', 'الاستخدام الآمن', 'الري الحديث', 'الري بالتنقيط',
            'الري المحوري', 'ترشيد استهلاك المياه', 'حماية مصادر المياه', 'المياه الجوفية',
            'الينابيع', 'الآبار', 'العيون', 'الأفلاج', 'السواقي', 'شبكات الري', 'قنوات الري',
            'water', 'clean water', 'sanitation', 'sewage', 'purification', 'dams', 'wells', 'irrigation'
        ]
    },
    7: {
        'name': 'الطاقة النظيفة',
        'keywords': [
            'طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'شبكة كهرباء', 'محطات توليد',
            'الطاقة الشمسية', 'الألواح الشمسية', 'الخلايا الشمسية', 'محطات شمسية', 'الطاقة الريحية',
            'توربينات الرياح', 'مزارع الرياح', 'الطاقة النووية', 'المفاعلات النووية', 'الطاقة المائية',
            'السدود الكهرومائية', 'الطاقة الكهرومائية', 'الطاقة الحرارية', 'الطاقة الحرارية الأرضية',
            'الطاقة الحيوية', 'الوقود الحيوي', 'الكتلة الحيوية', 'الهيدروجين الأخضر', 'الأمونيا الخضراء',
            'توليد الكهرباء', 'نقل الكهرباء', 'شبكات النقل', 'محطات التحويل', 'التوزيع الكهربائي',
            'العدادات الذكية', 'كفاءة الطاقة', 'ترشيد الطاقة', 'الطاقة المستدامة', 'الطاقة النظيفة',
            'الاستدامة الطاقوية', 'مزيج الطاقة', 'التحول الطاقوي', 'الحياد الكربوني',
            'energy', 'electricity', 'solar', 'renewable', 'wind', 'clean energy', 'power plant'
        ]
    },
    8: {
        'name': 'العمل اللائق',
        'keywords': [
            'عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'بطالة', 'مهارات مهنية',
            'العمل اللائق', 'الوظائف اللائقة', 'ظروف العمل', 'بيئة العمل', 'حقوق العمال',
            'النقابات العمالية', 'العمال', 'الموظفين', 'العمالة الوافدة', 'العمالة المنزلية',
            'العمالة الموسمية', 'العمل الحر', 'العمل عن بعد', 'المرونة الوظيفية', 'الأجور',
            'الرواتب', 'الحد الأدنى للأجور', 'التأمينات الاجتماعية', 'التقاعد', 'الضمان الاجتماعي',
            'السلامة المهنية', 'الصحة المهنية', 'الإصابات العمل', 'حوادث العمل', 'التدريب المهني',
            'التأهيل الوظيفي', 'التطوير المهني', 'المهارات الوظيفية', 'الكفاءات',
            'الاقتصاد غير الرسمي', 'العمالة غير المنتظمة', 'العمالة الناقصة', 'البطالة المقنعة',
            'employment', 'jobs', 'work', 'labor', 'workforce', 'decent work', 'unemployment'
        ]
    },
    9: {
        'name': 'الصناعة والابتكار',
        'keywords': [
            'صناعة', 'ابتكار', 'بنية تحتية', 'طرق', 'جسور', 'مصانع', 'تكنولوجيا',
            'القطاع الصناعي', 'المدن الصناعية', 'المناطق الصناعية', 'المصانع', 'المنشآت الصناعية',
            'التصنيع', 'الإنتاج الصناعي', 'المواد الخام', 'المواد الأولية', 'السلع المصنعة',
            'الابتكار التقني', 'الابتكار التكنولوجي', 'البحث والتطوير', 'الاختراعات', 'براءات الاختراع',
            'الملكية الفكرية', 'التقنيات الحديثة', 'التقنيات الناشئة', 'التحول الرقمي', 'الذكاء الاصطناعي',
            'إنترنت الأشياء', 'الروبوتات', 'الأتمتة', 'الصناعة 4.0', 'الثورة الصناعية الرابعة',
            'الطرق السريعة', 'الجسور', 'الأنفاق', 'الموانئ', 'المطارات', 'السكك الحديدية', 'المترو',
            'القطارات', 'شبكات النقل', 'اللوجستيات', 'سلاسل الإمداد', 'الخدمات اللوجستية',
            'industry', 'innovation', 'infrastructure', 'roads', 'bridges', 'factories', 'technology'
        ]
    },
    10: {
        'name': 'الحد من عدم المساواة',
        'keywords': [
            'مساواة', 'شمولية', 'فئات مهمشة', 'ذوي احتياجات خاصة', 'تمكين',
            'عدم المساواة', 'الفجوة الاجتماعية', 'الفجوة الاقتصادية', 'الفجوة الرقمية',
            'المناطق المهمشة', 'الأرياف', 'النائية', 'المناطق الحدودية', 'سكان البادية',
            'البدو', 'الرحل', 'اللاجئين', 'النازحين', 'المهاجرين', 'الأقليات',
            'ذوو الإعاقة', 'المعاقين', 'المكفوفين', 'الصم', 'ذوي الهمم', 'الاحتياجات الخاصة',
            'كبار السن', 'المسنين', 'المسنات', 'المتقاعدين', 'الأحداث', 'الأطفال',
            'الشمول المالي', 'الشمول الاجتماعي', 'الاندماج الاجتماعي', 'التكامل المجتمعي',
            'التماسك الاجتماعي', 'العدالة الاجتماعية', 'تكافؤ الفرص', 'المساواة في الحقوق',
            'inequality', 'inclusion', 'marginalized', 'disabled', 'refugees', 'social justice'
        ]
    },
    11: {
        'name': 'مدن مستدامة',
        'keywords': [
            'مدن', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل عام', 'بنية تحتية حضرية',
            'المدن المستدامة', 'المجتمعات المستدامة', 'المدن الذكية', 'التخطيط العمراني',
            'التطوير الحضري', 'التجديد الحضري', 'تطوير المدن', 'المناطق الحضرية',
            'المراكز الحضرية', 'المدن الكبرى', 'المدن الجديدة', 'المجتمعات العمرانية',
            'الإسكان الميسر', 'الإسكان الاجتماعي', 'الإسكان التنموي', 'الإسكان الشعبي',
            'الأحياء السكنية', 'المرافق العامة', 'الخدمات البلدية', 'النظافة', 'الإنارة العامة',
            'الطرق الداخلية', 'الأرصفة', 'الحدائق العامة', 'المتنزهات', 'المساحات الخضراء',
            'المواصلات العامة', 'الحافلات', 'التاكسي', 'مترو الأنفاق', 'الترام', 'القطار الكهربائي',
            'إدارة المخلفات', 'تدوير النفايات', 'النفايات الصلبة', 'المخلفات',
            'sustainable cities', 'urban planning', 'housing', 'transportation', 'public transport'
        ]
    },
    12: {
        'name': 'استهلاك مسؤول',
        'keywords': [
            'استهلاك', 'إنتاج', 'استدامة', 'كفاءة موارد', 'إعادة تدوير',
            'الاستهلاك المستدام', 'الإنتاج المستدام', 'أنماط الاستهلاك', 'ترشيد الاستهلاك',
            'الاستهلاك المسؤول', 'الاستهلاك الواعي', 'الاستهلاك الأخضر', 'المنتجات المستدامة',
            'المواد المستدامة', 'المواد المعاد تدويرها', 'إعادة التدوير', 'تدوير المخلفات',
            'إعادة الاستخدام', 'تقليل الاستهلاك', 'تقليل الهدر', 'هدر الطعام', 'الفاقد الغذائي',
            'الاقتصاد الدائري', 'التصميم المستدام', 'التعبئة المستدامة', 'المنتجات الصديقة للبيئة',
            'البصمة البيئية', 'البصمة الكربونية', 'استدامة الموارد', 'كفاءة الموارد',
            'كفاءة الطاقة', 'كفاءة المياه', 'ترشيد الطاقة', 'ترشيد المياه',
            'responsible consumption', 'sustainable production', 'recycling', 'circular economy'
        ]
    },
    13: {
        'name': 'العمل المناخي',
        'keywords': [
            'مناخ', 'تغير مناخي', 'انبعاثات', 'احتباس حراري', 'كربون',
            'التغير المناخي', 'تغيرات المناخ', 'الاحتباس الحراري', 'الاحترار العالمي',
            'غازات الاحتباس الحراري', 'غازات الدفيئة', 'انبعاثات الكربون', 'الكربون',
            'ثاني أكسيد الكربون', 'الميثان', 'الغازات', 'الانبعاثات الكربونية',
            'الحياد الكربوني', 'صفر كربون', 'خفض الانبعاثات', 'تخفيض الانبعاثات',
            'التخفيف من التغير المناخي', 'التكيف مع التغير المناخي', 'مقاومة المناخ',
            'الطقس المتطرف', 'الظواهر الجوية', 'الكوارث الطبيعية', 'الفيضانات', 'الجفاف',
            'العواصف', 'الأعاصير', 'حرائق الغابات', 'ارتفاع منسوب البحر', 'ذوبان الجليد',
            'الطاقة النظيفة', 'الطاقة المتجددة', 'الحلول المناخية', 'الاستدامة البيئية',
            'climate', 'climate change', 'emissions', 'carbon', 'global warming', 'net zero'
        ]
    },
    14: {
        'name': 'الحياة تحت الماء',
        'keywords': [
            'بحار', 'محيطات', 'أسماك', 'سواحل', 'ثروة بحرية', 'صيد',
            'الحياة البحرية', 'الكائنات البحرية', 'النظم البيئية البحرية', 'الشعاب المرجانية',
            'المرجان', 'السلاحف البحرية', 'الثدييات البحرية', 'الحيتان', 'الدلافين',
            'الأسماك', 'المخزون السمكي', 'الثروة السمكية', 'مصائد الأسماك', 'الصيد الجائر',
            'الاستزراع السمكي', 'المزارع السمكية', 'الأحياء المائية', 'الربيان', 'الجمبري',
            'تلوث البحار', 'التلوث البحري', 'المخلفات البحرية', 'اللدائن البحرية', 'البلاستيك في المحيطات',
            'المحميات البحرية', 'المناطق البحرية المحمية', 'حماية السواحل', 'إدارة السواحل',
            'الاقتصاد الأزرق', 'الموارد البحرية', 'الموارد الساحلية', 'السياحة البحرية',
            'oceans', 'seas', 'marine life', 'fisheries', 'coral reefs', 'marine conservation'
        ]
    },
    15: {
        'name': 'الحياة في البر',
        'keywords': [
            'بيئة', 'غابات', 'تنوع أحيائي', 'محيات طبيعية', 'حيوانات', 'نباتات',
            'النظم البيئية الأرضية', 'النظم البيئية البرية', 'الغابات', 'الأحراش', 'البراري',
            'المحميات الطبيعية', 'المحميات البرية', 'المناطق المحمية', 'الحدائق الوطنية',
            'الحياة البرية', 'الحيوانات البرية', 'الطيور', 'الكائنات الفطرية', 'الكائنات المهددة بالانقراض',
            'الأنواع المهددة', 'الأنواع النادرة', 'التنوع الحيوي', 'التنوع البيولوجي',
            'الموارد الوراثية', 'النباتات', 'الأشجار', 'النباتات الطبيعية', 'الغطاء النباتي',
            'التصحر', 'مكافحة التصحر', 'الزحف العمراني', 'تدهور الأراضي', 'تآكل التربة',
            'إعادة التشجير', 'زراعة الغابات', 'التشجير', 'الاستدامة البيئية',
            'terrestrial ecosystems', 'forests', 'biodiversity', 'wildlife', 'conservation'
        ]
    },
    16: {
        'name': 'السلام والعدالة',
        'keywords': [
            'سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'سيادة القانون',
            'الأمن والسلام', 'الاستقرار', 'الأمن المجتمعي', 'الأمن الوطني', 'مكافحة الإرهاب',
            'التطرف', 'مكافحة الجريمة', 'الجرائم', 'النزاعات', 'حل النزاعات', 'الوساطة',
            'المصالحة', 'المصالحة الوطنية', 'بناء السلام', 'حفظ السلام', 'بعثات السلام',
            'العدالة الجنائية', 'العدالة الناجزة', 'العدالة الانتقالية', 'القضاء', 'المحاكم',
            'النيابة العامة', 'التحكيم', 'القضاء الإداري', 'المجلس القضائي',
            'الحوكمة الرشيدة', 'الإدارة الرشيدة', 'مكافحة الفساد', 'الشفافية', 'المساءلة',
            'سيادة القانون', 'حكم القانون', 'المؤسسات الحكومية', 'الإصلاح المؤسسي',
            'حقوق الإنسان', 'الحريات العامة', 'الحقوق المدنية', 'الحقوق السياسية',
            'peace', 'justice', 'institutions', 'governance', 'rule of law', 'anti-corruption'
        ]
    },
    17: {
        'name': 'الشراكات',
        'keywords': [
            'شراكات', 'تعاون دولي', 'تمويل', 'منح', 'قروض', 'مساعدات دولية',
            'الشراكات الدولية', 'الشراكات الإقليمية', 'الشراكات المحلية', 'التعاون المشترك',
            'التعاون الثنائي', 'التعاون متعدد الأطراف', 'المنظمات الدولية', 'الأمم المتحدة',
            'البنك الدولي', 'صندوق النقد الدولي', 'المانحين', 'الجهات المانحة', 'الدول المانحة',
            'المساعدات الإنمائية', 'المساعدات الإنسانية', 'المساعدات الفنية', 'الدعم الفني',
            'التمويل المشترك', 'التمويل الدولي', 'الاستثمار الأجنبي', 'الاستثمار المباشر',
            'القطاع الخاص', 'الاستثمار الخاص', 'المنظمات غير الحكومية', 'المجتمع المدني',
            'المؤسسات الخيرية', 'الأوقاف', 'الصناديق التنموية', 'الصناديق السيادية',
            'partnerships', 'international cooperation', 'funding', 'grants', 'loans', 'aid'
        ]
    }
}

# --- 4. تصنيف الأهداف حسب الأبعاد ---
SDG_DIMENSIONS = {
    'social': [1, 2, 3, 4, 5, 10, 11, 16],      # الأهداف الاجتماعية
    'economic': [8, 9, 12, 17],                   # الأهداف الاقتصادية
    'environmental': [6, 7, 13, 14, 15]           # الأهداف البيئية
}

# --- 5. دوال التحليل الذكي ---
def extract_sdgs_from_text(text):
    """استخراج أهداف التنمية المستدامة من النص باستخدام الكلمات المفتاحية"""
    if not text:
        return []
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    detected_sdgs = []
    matched_keywords = {i: [] for i in range(1, 18)}
    
    for sdg_num, sdg_info in SDG_KEYWORDS.items():
        for keyword in sdg_info['keywords']:
            if keyword in text:
                detected_sdgs.append(sdg_num)
                matched_keywords[sdg_num].append(keyword)
                break
    
    return list(set(detected_sdgs)), matched_keywords

def calculate_sdg_metrics(detected_sdgs):
    """حساب المقاييس الأربعة الرئيسية من الأهداف المستخرجة"""
    if not detected_sdgs:
        return {
            'sdg_count': 0,
            'social_ratio': 0,
            'economic_ratio': 0,
            'environmental_ratio': 0,
            'balance_score': 0,
            'dimensions': {'social': 0, 'economic': 0, 'environmental': 0},
            'integration_score': 0,
            'sdg_list': []
        }
    
    sdg_count = len(detected_sdgs)
    
    social_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['social'])
    economic_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['economic'])
    environmental_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['environmental'])
    
    social_ratio = (social_count / sdg_count) * 100 if sdg_count > 0 else 0
    economic_ratio = (economic_count / sdg_count) * 100 if sdg_count > 0 else 0
    environmental_ratio = (environmental_count / sdg_count) * 100 if sdg_count > 0 else 0
    
    # درجة التوازن
    target = 33.33
    deviations = abs(social_ratio - target) + abs(economic_ratio - target) + abs(environmental_ratio - target)
    balance_score = max(0, 100 - (deviations / 2))
    
    # درجة التكامل (عدد الأبعاد المغطاة)
    dimensions_covered = 0
    if social_count > 0: dimensions_covered += 1
    if economic_count > 0: dimensions_covered += 1
    if environmental_count > 0: dimensions_covered += 1
    integration_score = (dimensions_covered / 3) * 100
    
    return {
        'sdg_count': sdg_count,
        'social_ratio': social_ratio,
        'economic_ratio': economic_ratio,
        'environmental_ratio': environmental_ratio,
        'balance_score': balance_score,
        'integration_score': integration_score,
        'dimensions': {
            'social': social_count,
            'economic': economic_count,
            'environmental': environmental_count
        },
        'sdg_list': detected_sdgs
    }

def predict_success(metrics, models):
    """التنبؤ بنجاح المشروع"""
    if not models:
        return 0.5, 0, 0.5
    
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
    hybrid_prob = (config['weight_ann'] * ann_prob + config['weight_xgb'] * xgb_prob)
    success_pred = 1 if hybrid_prob >= config['threshold'] else 0
    
    return hybrid_prob, success_pred, xgb_prob

def get_project_trend(metrics):
    """تحديد توجه المشروع"""
    ratios = {
        'اجتماعي': metrics['social_ratio'],
        'اقتصادي': metrics['economic_ratio'],
        'بيئي': metrics['environmental_ratio']
    }
    max_dimension = max(ratios, key=ratios.get)
    
    if metrics['balance_score'] > 80:
        return "🏆 متوازن (تكامل عالي)"
    elif max_dimension == 'اجتماعي':
        return "👥 ذو توجه اجتماعي"
    elif max_dimension == 'اقتصادي':
        return "💰 ذو توجه اقتصادي"
    else:
        return "🌍 ذو توجه بيئي"

def calculate_risk_score(metrics, p_budget, p_ben):
    """حساب درجة المخاطرة"""
    risk_factors = []
    
    # مخاطرة قلة الأهداف
    if metrics['sdg_count'] < 3:
        risk_factors.append(20)
    elif metrics['sdg_count'] < 5:
        risk_factors.append(10)
    else:
        risk_factors.append(5)
    
    # مخاطرة عدم التوازن
    if metrics['balance_score'] < 50:
        risk_factors.append(25)
    elif metrics['balance_score'] < 70:
        risk_factors.append(15)
    else:
        risk_factors.append(5)
    
    # مخاطرة الميزانية مقابل المستفيدين
    if p_budget > 0 and p_ben > 0:
        cost_per_beneficiary = p_budget / p_ben
        if cost_per_beneficiary > 100000:
            risk_factors.append(20)
        elif cost_per_beneficiary > 50000:
            risk_factors.append(10)
        else:
            risk_factors.append(5)
    
    total_risk = sum(risk_factors) / len(risk_factors)
    return min(100, total_risk)

def get_recommendations(metrics, success_prob, risk_score):
    """توليد توصيات ذكية"""
    recommendations = []
    
    if success_prob < 0.4:
        recommendations.append("🔴 يحتاج المشروع إلى إعادة هيكلة جذرية")
    elif success_prob < 0.6:
        recommendations.append("🟡 يوجد مجال للتحسين في تصميم المشروع")
    else:
        recommendations.append("🟢 المشروع في المسار الصحيح")
    
    if metrics['sdg_count'] < 3:
        recommendations.append("📌 زيادة عدد أهداف التنمية المستدامة المستهدفة")
    
    if metrics['balance_score'] < 50:
        recommendations.append("⚖️ تحسين التوازن بين الأبعاد الثلاثة")
    
    if metrics['integration_score'] < 66:
        recommendations.append("🔄 تعزيز التكامل بين الأبعاد المختلفة")
    
    if risk_score > 60:
        recommendations.append("⚠️ وضع خطة لإدارة المخاطر العالية")
    
    return recommendations

# --- 6. تحميل النماذج ---
models = load_trained_models()

# --- 7. واجهة المستخدم (تصميم احترافي) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', 'Noto Sans Arabic', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* تحسين مظهر الحقول */
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        padding: 14px 16px !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:hover, .stTextArea textarea:hover {
        border-color: #94A3B8 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #0F172A !important;
        box-shadow: 0 0 0 3px rgba(15,23,42,0.1) !important;
    }
    
    /* تنسيق الحقول جنباً إلى جنب */
    .row-widget {
        margin-bottom: 0 !important;
    }
    
    /* زر التحليل */
    div.stButton > button {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        height: 56px !important;
        font-weight: 600 !important;
        border: none !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 16px rgba(15,23,42,0.2) !important;
        transition: all 0.3s ease !important;
        margin-top: 20px !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        box-shadow: 0 12px 24px rgba(15,23,42,0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* العنوان الرئيسي */
    h1 {
        color: #0F172A;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 40px;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* تسميات الحقول */
    label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 8px !important;
    }
    
    /* شارات SDG */
    .sdg-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 30px;
        margin: 6px;
        font-size: 0.95rem;
        font-weight: 500;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .sdg-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    /* بطاقات التحليل */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* تحسين المسافات */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* شريط التقدم */
    .stProgress > div > div {
        background-color: #0F172A !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 8. العنوان الرئيسي ---
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1>المنصة الذكية لتحليل المشاريع التنموية</h1>
        <p style='color: #64748B; font-size: 1.1rem; margin-top: -10px;'>تحليل دقيق باستخدام تقنيات الذكاء الاصطناعي</p>
    </div>
""", unsafe_allow_html=True)

# --- 9. نموذج الإدخال المحسن ---
with st.container():
    with st.form("professional_analysis_form"):
        # الصف الأول: اسم المشروع والمجال
        col1, col2 = st.columns([2, 1])
        with col1:
            p_name = st.text_input("📋 اسم المشروع", placeholder="أدخل اسم المشروع...")
        with col2:
            p_cat = st.selectbox("🏷️ مجال المشروع", ["", "تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
        
        # الصف الثاني: وصف المشروع (يمتد ليغطي المساحة)
        col3, col4 = st.columns([2, 1])
        with col3:
            p_desc = st.text_area("📝 وصف فكرة المشروع", height=180, 
                                 placeholder="أدخل تفاصيل المشروع هنا... (سيتم تحليل النص لاستخراج أهداف التنمية المستدامة)")
        with col4:
            st.markdown("<div style='margin-top: 0;'>", unsafe_allow_html=True)
            p_budget = st.number_input("💰 الميزانية المرصودة (SAR)", min_value=0, value=0, step=1000)
            p_ben = st.number_input("👥 عدد المستفيدين المتوقع", min_value=0, value=0, step=100)
            st.markdown("</div>", unsafe_allow_html=True)
        
        submit_btn = st.form_submit_button("🔮 تحليل المشروع بالذكاء الاصطناعي", use_container_width=True)

# --- 10. التحليل والنتائج ---
if submit_btn:
    if not p_name or not p_desc or not p_cat or p_budget == 0 or p_ben == 0:
        st.error("⚠️ يرجى تزويد المنصة بجميع البيانات المطلوبة لإتمام عملية التحليل.")
    else:
        with st.spinner('🧠 جاري تحليل المشروع واستخراج الأهداف...'):
            
            # استخراج الأهداف
            full_text = p_name + " " + p_desc
            detected_sdgs, matched_keywords = extract_sdgs_from_text(full_text)
            
            # حساب المقاييس
            metrics = calculate_sdg_metrics(detected_sdgs)
            
            # التنبؤ بالنجاح
            if models:
                success_prob, success_pred, xgb_prob = predict_success(metrics, models)
            else:
                success_prob, success_pred, xgb_prob = 0.5, 0, 0.5
            
            # حساب المخاطرة والتوصيات
            risk_score = calculate_risk_score(metrics, p_budget, p_ben)
            recommendations = get_recommendations(metrics, success_prob, risk_score)
            
            # تحديد توجه المشروع
            project_trend = get_project_trend(metrics)
            
            # حساب المؤشرات المالية
            sroi_val = round(success_prob * (p_ben / (p_budget/1000)) if p_budget > 0 else 0, 2)
            economic_impact = f"{int(p_budget * success_prob * 1.45):,}" if p_budget > 0 else "0"
            
            # --- 11. عرض النتائج المتقدمة ---
            
            # المؤشر الرئيسي
            col_main1, col_main2, col_main3 = st.columns(3)
            
            with col_main1:
                st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="color: #64748B; font-size: 0.9rem;">نسبة النجاح المتوقعة</div>
                        <div style="font-size: 3rem; font-weight: 800; color: #0F172A;">{success_prob*100:.1f}%</div>
                        <div style="margin-top: 10px;">
                            <span style="background: {'#22C55E' if success_pred == 1 else '#EF4444'}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">
                                {'✅ ناجح' if success_pred == 1 else '⚠️ غير ناجح'}
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_main2:
                st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="color: #64748B; font-size: 0.9rem;">درجة المخاطرة</div>
                        <div style="font-size: 3rem; font-weight: 800; color: #0F172A;">{risk_score:.1f}%</div>
                        <div style="margin-top: 10px;">
                            <span style="background: {'#22C55E' if risk_score < 30 else '#F59E0B' if risk_score < 60 else '#EF4444'}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">
                                {'منخفضة' if risk_score < 30 else 'متوسطة' if risk_score < 60 else 'مرتفعة'}
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_main3:
                st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <div style="color: #64748B; font-size: 0.9rem;">توجه المشروع</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-top: 15px;">{project_trend}</div>
                        <div style="margin-top: 15px; color: #64748B; font-size: 0.9rem;">{metrics['sdg_count']} أهداف مستهدفة</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # أهداف SDG
            st.markdown("""
                <div style="margin-top: 30px;">
                    <h3 style="color: #0F172A; font-size: 1.3rem; margin-bottom: 15px;">🎯 أهداف التنمية المستدامة المستخرجة</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if detected_sdgs:
                cols = st.columns(4)
                for i, sdg in enumerate(detected_sdgs):
                    with cols[i % 4]:
                        st.markdown(f"""
                            <div class="sdg-badge" style="width: 100%; text-align: center;">
                                الهدف {sdg}: {SDG_KEYWORDS[sdg]['name']}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("لم يتم العثور على أهداف محددة في النص")
            
            # تحليل الأبعاد
            st.markdown("""
                <div style="margin-top: 40px;">
                    <h3 style="color: #0F172A; font-size: 1.3rem; margin-bottom: 20px;">📊 تحليل الأبعاد التنموية</h3>
                </div>
            """, unsafe_allow_html=True)
            
            dim_cols = st.columns(3)
            
            with dim_cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600;">البعد الاجتماعي</span>
                            <span style="font-size: 1.5rem; font-weight: 700;">{metrics['dimensions']['social']}</span>
                        </div>
                        <div style="margin-top: 10px; height: 8px; background: #E2E8F0; border-radius: 10px;">
                            <div style="width: {metrics['social_ratio']}%; height: 100%; background: #0F172A; border-radius: 10px;"></div>
                        </div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.9rem;">{metrics['social_ratio']:.1f}% من الأهداف</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with dim_cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600;">البعد الاقتصادي</span>
                            <span style="font-size: 1.5rem; font-weight: 700;">{metrics['dimensions']['economic']}</span>
                        </div>
                        <div style="margin-top: 10px; height: 8px; background: #E2E8F0; border-radius: 10px;">
                            <div style="width: {metrics['economic_ratio']}%; height: 100%; background: #0F172A; border-radius: 10px;"></div>
                        </div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.9rem;">{metrics['economic_ratio']:.1f}% من الأهداف</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with dim_cols[2]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600;">البعد البيئي</span>
                            <span style="font-size: 1.5rem; font-weight: 700;">{metrics['dimensions']['environmental']}</span>
                        </div>
                        <div style="margin-top: 10px; height: 8px; background: #E2E8F0; border-radius: 10px;">
                            <div style="width: {metrics['environmental_ratio']}%; height: 100%; background: #0F172A; border-radius: 10px;"></div>
                        </div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.9rem;">{metrics['environmental_ratio']:.1f}% من الأهداف</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # مؤشرات متقدمة
            st.markdown("""
                <div style="margin-top: 40px;">
                    <h3 style="color: #0F172A; font-size: 1.3rem; margin-bottom: 20px;">📈 مؤشرات متقدمة</h3>
                </div>
            """, unsafe_allow_html=True)
            
            adv_cols = st.columns(3)
            
            with adv_cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem;">درجة التوازن</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{metrics['balance_score']:.1f}%</div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.85rem;">كلما زادت كان المشروع أكثر توازناً</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with adv_cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem;">درجة التكامل</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{metrics['integration_score']:.1f}%</div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.85rem;">نسبة تغطية الأبعاد الثلاثة</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with adv_cols[2]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color: #64748B; font-size: 0.9rem;">العائد الاجتماعي</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #0F172A;">{sroi_val}x</div>
                        <div style="margin-top: 8px; color: #64748B; font-size: 0.85rem;">لكل ريال مستثمر</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # التوصيات
            st.markdown("""
                <div style="margin-top: 40px;">
                    <h3 style="color: #0F172A; font-size: 1.3rem; margin-bottom: 20px;">💡 التوصيات الذكية</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"""
                    <div style="background: #F8FAFC; border-right: 4px solid #0F172A; padding: 15px; margin-bottom: 10px; border-radius: 8px;">
                        {rec}
                    </div>
                """, unsafe_allow_html=True)
            
            # الخلاصة
            st.markdown(f"""
                <div style="margin-top: 40px; background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 30px; border-radius: 16px; color: white;">
                    <h4 style="color: white; margin-bottom: 15px;">📋 الخلاصة التنفيذية</h4>
                    <p style="line-height: 1.8; margin-bottom: 10px;">
                        المشروع <strong>"{p_name}"</strong> في مجال <strong>{p_cat}</strong> يستهدف <strong>{metrics['sdg_count']}</strong> 
                        من أهداف التنمية المستدامة. تبلغ نسبة النجاح المتوقعة <strong>{success_prob*100:.1f}%</strong> 
                        مع درجة مخاطرة <strong>{risk_score:.1f}%</strong>.
                    </p>
                    <p style="line-height: 1.8;">
                        {'✅ يوصى بالمضي قدماً في المشروع مع التركيز على نقاط القوة المكتشفة.' if success_pred == 1 else '⚠️ يوصى بإعادة تقييم المشروع وتحسين التكامل بين الأبعاد المختلفة قبل التنفيذ.'}
                    </p>
                </div>
            """, unsafe_allow_html=True)

# --- 12. التذييل ---
st.markdown("""
    <div style='text-align: center; color: #94A3B8; font-size: 0.9rem; margin-top: 80px; padding: 20px; border-top: 1px solid #E2E8F0;'>
        © جميع الحقوق محفوظة - المنصة الذكية لتحليل المشاريع التنموية 2024
    </div>
""", unsafe_allow_html=True)
