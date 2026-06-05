import streamlit as st
import cv2
import torch
import numpy as np
import yaml
import torchvision
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetClassificationHead
import torchvision.transforms.functional as F

from src.network import CheatingLSTM
from src.pose import get_pose_features

st.set_page_config(page_title="Monitor SIS421 USFX", layout="wide")
st.title("[SISTEMA] Centro de Monitoreo e Inferencia en Examenes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_production_models():
    phone_id, paper_id = 0, 1
    try:
        with open("data/roboflow_data/data.yaml", "r") as f:
            config = yaml.safe_load(f)
        for idx, name in enumerate(config["names"]):
            name_lower = name.lower()
            if "phone" in name_lower or "celular" in name_lower: phone_id = idx
            if "paper" in name_lower or "papel" in name_lower: paper_id = idx
    except: pass

    # Inicializar RetinaNet v2
    retina_model = retinanet_resnet50_fpn_v2(weights=None)
    in_features = retina_model.head.classification_head.conv[0][0].in_channels
    num_anchors = retina_model.head.classification_head.num_anchors
    retina_model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features, num_anchors=num_anchors, num_classes=3
    )
    try: retina_model.load_state_dict(torch.load("models/retinanet.pth", map_location=device))
    except: print("Aviso: models/retinanet.pth no encontrado.")
    retina_model.to(device).eval()

    lstm_model = CheatingLSTM()
    try: lstm_model.load_state_dict(torch.load("models/best_lstm.pth", map_location=device))
    except: print("Aviso: Usando pesos base para la LSTM.")
    lstm_model.to(device).eval()
    
    return retina_model, lstm_model, phone_id, paper_id

retina_net, lstm_net, phone_idx, paper_idx = load_production_models()

st.sidebar.subheader("Configuracion de Entrada")
source_type = st.sidebar.selectbox("Selecciona la Fuente de Video", ["Camara Web Local", "Archivo de Video (.mp4)"])
source_input = 0 if source_type == "Camara Web Local" else "data/test_video.mp4"

threshold = st.sidebar.slider("Umbral de Alerta", 0.40, 0.90, 0.65)
run_pipeline = st.sidebar.checkbox("Iniciar Monitoreo", value=False)

col_video, col_dash = st.columns([2, 1])
with col_video: VIDEO_CONTAINER = st.empty()
with col_dash:
    bar = st.progress(0.0)
    status_text = st.empty()

if run_pipeline:
    cap = cv2.VideoCapture(source_input)
    
    lista_temporal = []
    contador_cuadros = 0
    p_phone, p_paper = 0.0, 0.0
    cajas_cache, scores_cache, labels_cache = [], [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plotted_rgb = frame_rgb.copy()
        
        if contador_cuadros % 3 == 0:
            img_tensor = F.to_tensor(frame_rgb).to(device)
            with torch.no_grad():
                retina_out = retina_net([img_tensor])[0]
            cajas_cache = retina_out['boxes'].cpu().numpy()
            scores_cache = retina_out['scores'].cpu().numpy()
            labels_cache = retina_out['labels'].cpu().numpy()
            p_phone, p_paper = 0.0, 0.0
            
        contador_cuadros += 1
        
        for box, score, label in zip(cajas_cache, scores_cache, labels_cache):
            if score > 0.3: 
                c_id = label - 1 
                if c_id == phone_idx: p_phone = max(p_phone, score)
                if c_id == paper_idx: p_paper = max(p_paper, score)
                
                xmin, ymin, xmax, ymax = map(int, box)
                color = (255, 0, 0) if c_id == phone_idx else (0, 255, 0)
                cv2.rectangle(plotted_rgb, (xmin, ymin), (xmax, ymax), color, 2)
            
        pose_out = get_pose_features(frame_rgb)
        
        vector_cuadro = np.concatenate(([p_phone, p_paper], pose_out))
        lista_temporal.append(vector_cuadro)
        
        if len(lista_temporal) > 30:
            lista_temporal.pop(0)
            
        if len(lista_temporal) == 30:
            matriz_secuencia = np.array(lista_temporal)
            input_tensor = torch.tensor(matriz_secuencia, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                score_lstm = lstm_net(input_tensor).item()
        else:
            score_lstm = 0.0
            
        bar.progress(float(score_lstm))
        alto, ancho, _ = frame.shape
        
        if score_lstm >= threshold:
            status_text.error(f"[ALERTA] ACCION SOSPECHOSA DETECTADA ({score_lstm*100:.1f}%)")
            cv2.rectangle(plotted_rgb, (0, 0), (ancho, alto), (255, 0, 0), 10)
        else:
            status_text.success(f"[OK] CONTROL: NORMAL ({score_lstm*100:.1f}%)")
            
        VIDEO_CONTAINER.image(plotted_rgb)
    cap.release()