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
