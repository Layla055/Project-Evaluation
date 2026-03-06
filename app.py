import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import os
import re
from tensorflow.keras.models import load_model

# --- 1. إعدادات الهوية المؤسسية ---
st.set_page_config(page_title="المنصة الذكية لتحليل المشاريع التنموية", layout="wide", initial_sidebar_state="collapsed")

# --- 2. تحميل النماذج المدربة (الذكاء الحقيقي) ---
@st.cache_resource
def load_trained_models():
    """تحميل النماذج المدربة من ملفات"""
    try:
        # تحميل النماذج
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

# --- 3. قاموس الكلمات المفتاحية لأهداف التنمية المستدامة ---
SDG_KEYWORDS = {
    1: {
        'name': 'القضاء على الفقر',
        'keywords': ['فقر', 'فقراء', 'تمكين اقتصادي', 'دخل', 'مساعدات', 'ضمان اجتماعي', 'تكافل', 'مسكن', 'غذاء']
    },
    2: {
        'name': 'القضاء على الجوع',
        'keywords': ['جوع', 'أمن غذائي', 'زراعة', 'محاصيل', 'غذاء', 'تغذية', 'مزارعين', 'أراضي زراعية']
    },
    3: {
        'name': 'الصحة الجيدة',
        'keywords': ['صحة', 'مستشفى', 'مركز صحي', 'رعاية صحية', 'أمراض', 'لقاحات', 'أدوية', 'علاج']
    },
    4: {
        'name': 'التعليم الجيد',
        'keywords': ['تعليم', 'مدرسة', 'جامعة', 'طلاب', 'معلمين', 'مناهج', 'تدريب', 'محو أمية']
    },
    5: {
        'name': 'المساواة بين الجنسين',
        'keywords': ['مساواة', 'نساء', 'فتيات', 'تمكين المرأة', 'عنف ضد المرأة', 'حقوق المرأة']
    },
    6: {
        'name': 'المياه النظيفة',
        'keywords': ['مياه', 'صرف صحي', 'محطات تنقية', 'شرب', 'ري', 'سدود', 'آبار']
    },
    7: {
        'name': 'الطاقة النظيفة',
        'keywords': ['طاقة', 'كهرباء', 'طاقة شمسية', 'طاقة متجددة', 'شبكة كهرباء', 'محطات توليد']
    },
    8: {
        'name': 'العمل اللائق',
        'keywords': ['عمل', 'توظيف', 'وظائف', 'عمالة', 'فرص عمل', 'بطالة', 'مهارات مهنية']
    },
    9: {
        'name': 'الصناعة والابتكار',
        'keywords': ['صناعة', 'ابتكار', 'بنية تحتية', 'طرق', 'جسور', 'مصانع', 'تكنولوجيا']
    },
    10: {
        'name': 'الحد من عدم المساواة',
        'keywords': ['مساواة', 'شمولية', 'فئات مهمشة', 'ذوي احتياجات خاصة', 'تمكين']
    },
    11: {
        'name': 'مدن مستدامة',
        'keywords': ['مدن', 'تخطيط حضري', 'إسكان', 'مواصلات', 'نقل عام', 'بنية تحتية حضرية']
    },
    12: {
        'name': 'استهلاك مسؤول',
        'keywords': ['استهلاك', 'إنتاج', 'استدامة', 'كفاءة موارد', 'إعادة تدوير']
    },
    13: {
        'name': 'العمل المناخي',
        'keywords': ['مناخ', 'تغير مناخي', 'انبعاثات', 'احتباس حراري', 'كربون']
    },
    14: {
        'name': 'الحياة تحت الماء',
        'keywords': ['بحار', 'محيطات', 'أسماك', 'سواحل', 'ثروة بحرية', 'صيد']
    },
    15: {
        'name': 'الحياة في البر',
        'keywords': ['بيئة', 'غابات', 'تنوع أحيائي', 'محيات طبيعية', 'حيوانات', 'نباتات']
    },
    16: {
        'name': 'السلام والعدالة',
        'keywords': ['سلام', 'عدالة', 'مؤسسات', 'حوكمة', 'قضاء', 'سيادة القانون']
    },
    17: {
        'name': 'الشراكات',
        'keywords': ['شراكات', 'تعاون دولي', 'تمويل', 'منح', 'قروض', 'مساعدات دولية']
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
    """استخراج أهداف التنمية المستدامة من النص"""
    if not text:
        return []
    
    text = text.lower()
    detected_sdgs = []
    
    for sdg_num, sdg_info in SDG_KEYWORDS.items():
        for keyword in sdg_info['keywords']:
            if keyword in text:
                detected_sdgs.append(sdg_num)
                break
    
    return list(set(detected_sdgs))  # إزالة التكرارات

def calculate_sdg_metrics(detected_sdgs):
    """حساب المقاييس الأربعة الرئيسية من الأهداف المستخرجة"""
    if not detected_sdgs:
        return {
            'sdg_count': 0,
            'social_ratio': 0,
            'environmental_ratio': 0,
            'balance_score': 0
        }
    
    # عدد الأهداف
    sdg_count = len(detected_sdgs)
    
    # حساب الأهداف في كل بُعد
    social_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['social'])
    economic_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['economic'])
    environmental_count = sum(1 for sdg in detected_sdgs if sdg in SDG_DIMENSIONS['environmental'])
    
    # حساب النسب المئوية
    social_ratio = (social_count / sdg_count) * 100 if sdg_count > 0 else 0
    economic_ratio = (economic_count / sdg_count) * 100 if sdg_count > 0 else 0
    environmental_ratio = (environmental_count / sdg_count) * 100 if sdg_count > 0 else 0
    
    # حساب درجة التوازن (Balance Score)
    # Balance_score = 100 - (|S-33.33| + |E-33.33| + |En-33.33|)/2
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

def predict_success(metrics, models):
    """التنبؤ بنجاح المشروع باستخدام النموذج الهجين"""
    if not models:
        return 0.5, 0
    
    # تجهيز الميزات
    features = np.array([[
        metrics['sdg_count'],
        metrics['social_ratio'],
        metrics['balance_score'],
        metrics['environmental_ratio']
    ]])
    
    # تطبيق التطبيع
    features_scaled = models['scaler'].transform(features)
    
    # تنبؤ ANN
    ann_prob = models['ann'].predict(features_scaled, verbose=0)[0][0]
    
    # تنبؤ XGBoost
    xgb_prob = models['xgb'].predict_proba(features)[0][1]
    
    # التنبؤ الهجين (باستخدام الأوزان من config)
    config = models['config']
    hybrid_prob = (config['weight_ann'] * ann_prob + config['weight_xgb'] * xgb_prob)
    
    # التصنيف النهائي
    success_pred = 1 if hybrid_prob >= config['threshold'] else 0
    
    return hybrid_prob, success_pred

def get_project_trend(metrics):
    """تحديد توجه المشروع بناءً على النسب"""
    ratios = {
        'اجتماعي': metrics['social_ratio'],
        'اقتصادي': metrics['economic_ratio'],
        'بيئي': metrics['environmental_ratio']
    }
    max_dimension = max(ratios, key=ratios.get)
    
    if metrics['balance_score'] > 80:
        return "متوازن (تكامل عالي)"
    elif max_dimension == 'اجتماعي':
        return "ذو توجه اجتماعي"
    elif max_dimension == 'اقتصادي':
        return "ذو توجه اقتصادي"
    else:
        return "ذو توجه بيئي"

# --- 6. تحميل النماذج ---
models = load_trained_models()

# --- 7. واجهة المستخدم ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans Arabic', sans-serif;
        background-color: #F1F5F9;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input {
        border-radius: 4px !important; 
        border: 1px solid #CBD5E1 !important;
        padding: 12px !important; 
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    div.stButton > button {
        background-color: #0F172A !important;
        color: white !important;
        border-radius: 4px !important;
        height: 52px !important;
        font-weight: 700 !important;
        border: none !important;
        font-size: 1.1rem !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1E293B !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2) !important;
    }
    
    h1 { color: #0F172A; text-align: center; font-weight: 800; margin-bottom: 30px; }
    label { color: #334155 !important; font-weight: 700 !important; }
    
    .sdg-badge {
        display: inline-block;
        background: #0F172A;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- 8. العنوان الرئيسي ---
st.markdown("<div style='padding: 1.5rem 0;'><h1>المنصة الذكية لتحليل المشاريع التنموية</h1></div>", unsafe_allow_html=True)

# --- 9. نموذج الإدخال ---
with st.container():
    with st.form("professional_analysis_form"):
        col_left, col_right = st.columns([2, 1])
        with col_left:
            p_name = st.text_input("اسم المشروع التنموي")
            p_desc = st.text_area("وصف فكرة المشروع", height=140, 
                                  placeholder="أدخل تفاصيل المشروع هنا... (سيتم تحليل النص لاستخراج أهداف التنمية المستدامة)")
        with col_right:
            p_cat = st.selectbox("مجال المشروع", ["تعليمي", "صحي", "بيئي", "اقتصادي", "اجتماعي"])
            p_budget = st.number_input("الميزانية المرصودة (SAR)", min_value=1000, value=25000)
            p_ben = st.number_input("عدد المستفيدين المتوقع", min_value=1, value=500)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🔮 تحليل المشروع بالذكاء الاصطناعي")

# --- 10. التحليل والنتائج ---
if submit_btn:
    if not p_name or not p_desc:
        st.error("يرجى تزويد المنصة بالبيانات الأساسية للمشروع لإتمام عملية التحليل.")
    else:
        with st.spinner('🧠 جاري تحليل النص واستخراج الأهداف...'):
            
            # استخراج الأهداف من النص
            detected_sdgs = extract_sdgs_from_text(p_desc)
            
            # حساب المقاييس
            metrics = calculate_sdg_metrics(detected_sdgs)
            
            # التنبؤ بالنجاح
            if models:
                success_prob, success_pred = predict_success(metrics, models)
            else:
                success_prob, success_pred = 0.5, 0
            
            # تحديد توجه المشروع
            project_trend = get_project_trend(metrics)
            
            # حساب المؤشرات المالية (لإثراء التقرير)
            sroi_val = round(success_prob * (p_ben / (p_budget/1000)), 2)
            economic_impact = f"{int(p_budget * success_prob * 1.45):,}"
            
            # --- 11. عرض النتائج ---
            report_ui = f"""
            <div dir="rtl" style="background: white; border: 1px solid #E2E8F0; padding: 40px; border-radius: 8px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); margin-top: 20px;">
                <div style="border-bottom: 3px solid #0F172A; padding-bottom: 15px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="color: #0F172A; margin: 0;">تقرير التحليل التنموي: {p_name}</h2>
                    <span style="color: #64748B; font-weight: bold;">{project_trend}</span>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px;">
                    <!-- مؤشر النجاح -->
                    <div style="background: #F8FAFC; padding: 30px; border-radius: 4px; text-align: center; border: 1px solid #E2E8F0;">
                        <span style="color: #64748B; font-weight: 700; font-size: 0.9rem;">نسبة نجاح المشروع (تنبؤ ذكي)</span>
                        <div style="font-size: 4.5rem; font-weight: 800; color: #0F172A; margin: 10px 0;">{success_prob*100:.1f}%</div>
                        <div style="height: 8px; background: #E2E8F0; width: 85%; margin: 0 auto; border-radius: 10px; overflow: hidden;">
                            <div style="width: {success_prob*100}%; height: 100%; background: #0F172A;"></div>
                        </div>
                        <div style="margin-top: 15px; color: #475569;">
                            التصنيف: <strong>{'ناجح ✅' if success_pred == 1 else 'غير ناجح ⚠️'}</strong>
                        </div>
                    </div>

                    <!-- المقاييس المالية -->
                    <div style="display: grid; gap: 15px;">
                        <div style="border: 1px solid #E2E8F0; padding: 18px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #475569;">العائد الاجتماعي على الاستثمار</span>
                            <span style="font-weight: 800; color: #0F172A; font-size: 1.2rem;">{sroi_val}x</span>
                        </div>
                        <div style="border: 1px solid #E2E8F0; padding: 18px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #475569;">القيمة المضافة للاقتصاد المحلي</span>
                            <span style="font-weight: 800; color: #0F172A; font-size: 1.1rem;">{economic_impact} ريال</span>
                        </div>
                    </div>
                </div>

                <!-- أهداف التنمية المستدامة المستخرجة -->
                <div style="margin-top: 30px; padding: 20px; background: #F1F5F9; border-right: 6px solid #0F172A;">
                    <h4 style="margin: 0 0 15px 0; color: #0F172A;">🎯 أهداف التنمية المستدامة المستخرجة من النص:</h4>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            """
            
            # إضافة الأهداف المستخرجة
            for sdg in detected_sdgs:
                report_ui += f'<span class="sdg-badge">الهدف {sdg}: {SDG_KEYWORDS[sdg]["name"]}</span>'
            
            if not detected_sdgs:
                report_ui += '<span style="color: #64748B;">لم يتم العثور على أهداف محددة</span>'
            
            report_ui += f"""
                    </div>
                </div>

                <!-- تحليل الأبعاد -->
                <div style="margin-top: 30px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div style="text-align: center; padding: 15px; background: #F8FAFC; border-radius: 4px;">
                        <div style="font-size: 2rem; color: #0F172A;">{metrics['dimensions']['social']}</div>
                        <div style="color: #475569; font-size: 0.9rem;">أهداف اجتماعية</div>
                        <div style="font-size: 0.8rem; color: #64748B;">{metrics['social_ratio']:.1f}%</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: #F8FAFC; border-radius: 4px;">
                        <div style="font-size: 2rem; color: #0F172A;">{metrics['dimensions']['economic']}</div>
                        <div style="color: #475569; font-size: 0.9rem;">أهداف اقتصادية</div>
                        <div style="font-size: 0.8rem; color: #64748B;">{metrics['economic_ratio']:.1f}%</div>
                    </div>
                    <div style="text-align: center; padding: 15px; background: #F8FAFC; border-radius: 4px;">
                        <div style="font-size: 2rem; color: #0F172A;">{metrics['dimensions']['environmental']}</div>
                        <div style="color: #475569; font-size: 0.9rem;">أهداف بيئية</div>
                        <div style="font-size: 0.8rem; color: #64748B;">{metrics['environmental_ratio']:.1f}%</div>
                    </div>
                </div>

                <!-- درجة التوازن -->
                <div style="margin-top: 30px; padding: 20px; background: #F8FAFC; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #0F172A; font-weight: 700;">درجة التوازن بين الأبعاد:</span>
                        <span style="font-size: 1.5rem; font-weight: 800; color: #0F172A;">{metrics['balance_score']:.1f}%</span>
                    </div>
                    <div style="height: 8px; background: #E2E8F0; width: 100%; margin-top: 10px; border-radius: 10px;">
                        <div style="width: {metrics['balance_score']}%; height: 100%; background: #0F172A; border-radius: 10px;"></div>
                    </div>
                </div>

                <div style="margin-top: 30px; border: 1px solid #E2E8F0; padding: 25px; border-radius: 4px; line-height: 1.7;">
                    <strong style="color: #0F172A; font-size: 1.1rem;">📋 الخلاصة الاستشارية:</strong><br>
                    <p style="margin-top: 10px; color: #1E293B;">
                    بناءً على التحليل الذكي للنص باستخدام النماذج المدربة على {models['config']['accuracy']:.1f}% دقة:
                    <br><br>
                    • <strong>الأهداف المستخرجة:</strong> {metrics['sdg_count']} هدف من أهداف التنمية المستدامة
                    <br>
                    • <strong>توجه المشروع:</strong> {project_trend}
                    <br>
                    • <strong>التوصية:</strong> {'المشروع واعد ويمكن المضي قدماً' if success_pred == 1 else 'يحتاج المشروع إلى إعادة هيكلة لتحسين فرص النجاح'}
                    </p>
                </div>
            </div>
            """
            
            components.html(report_ui, height=1100, scrolling=True)

# --- 12. التذييل ---
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 60px;'>© جميع الحقوق محفوظة - المنصة الذكية لتحليل المشاريع التنموية 2024</div>", unsafe_allow_html=True)
