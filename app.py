import streamlit as st
import cv2
import torch
import numpy as np
import yaml
import torchvision
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetClassificationHead
import torchvision.transforms.functional as F
import mediapipe as mp
import time

from src.network import CheatingLSTM
from src.pose import pose_handler, mp_pose

mp_dibujo = mp.solutions.drawing_utils

st.set_page_config(page_title="control de examenes", layout="wide")
st.title("sistema de monitoreo de examenes en linea")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_production_models():
    phone_id = 0
    paper_id = 1

    retina_model = retinanet_resnet50_fpn_v2(weights=None)
    in_features = retina_model.head.classification_head.conv[0][0].in_channels
    num_anchors = retina_model.head.classification_head.num_anchors
    
    retina_model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features, num_anchors=num_anchors, num_classes=2
    )
    
    retina_model.load_state_dict(torch.load("models/retinanet.pth", map_location=device))
    retina_model.to(device).eval()

    lstm_model = CheatingLSTM()
    lstm_model.load_state_dict(torch.load("models/best_lstm.pth", map_location=device))
    lstm_model.to(device).eval()
    
    return retina_model, lstm_model, phone_id, paper_id

retina_net, lstm_net, phone_idx, paper_idx = load_production_models()

st.sidebar.subheader("configuracion")
source_type = st.sidebar.selectbox("fuente de video", ["webcam local", "video de prueba (.mp4)"])
source_input = 0 if source_type == "webcam local" else "data/test_video.mp4"

threshold = st.sidebar.slider("umbral de sensibilidad", 0.40, 0.90, 0.65)
run_pipeline = st.sidebar.checkbox("iniciar control", value=False)

st.sidebar.subheader("telemetria en bruto del modelo")
debug_panel = st.sidebar.empty()

col_video, col_dash = st.columns([2, 1])
with col_video: VIDEO_CONTAINER = st.empty()
with col_dash:
    bar = st.progress(0.0)
    status_text = st.empty()

if run_pipeline:
    cap = cv2.VideoCapture(source_input)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    lista_temporal = []
    contador_cuadros = 0
    cajas_cache, scores_cache, labels_cache = [], [], []
    score_lstm = 0.0
    
    SALTAR_RETINA = 4 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plotted_rgb = frame_rgb.copy()
        
        resultados_pose = pose_handler.process(frame_rgb)
        
        angle, dist, offset_x = 0.0, 0.5, 0.0
        if resultados_pose.pose_landmarks:
            mp_dibujo.draw_landmarks(plotted_rgb, resultados_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            puntos = resultados_pose.pose_landmarks.landmark
            nose = np.array([puntos[mp_pose.PoseLandmark.NOSE].x, puntos[mp_pose.PoseLandmark.NOSE].y])
            left_shoulder = np.array([puntos[mp_pose.PoseLandmark.LEFT_SHOULDER].x, puntos[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            right_shoulder = np.array([puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, puntos[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
            
            mid_shoulder = (left_shoulder + right_shoulder) / 2.0
            neck_vector = nose - mid_shoulder
            
            angle = np.arctan2(neck_vector[1], neck_vector[0])
            dist = np.linalg.norm(left_shoulder - right_shoulder)
            offset_x = neck_vector[0]
            
        pose_out = np.array([angle, dist, offset_x])
        
        if contador_cuadros % SALTAR_RETINA == 0:
            img_tensor = F.to_tensor(frame_rgb).to(device)
            with torch.no_grad():
                retina_out = retina_net([img_tensor])[0]
                
            cajas_cache = retina_out['boxes'].cpu().numpy()
            scores_cache = retina_out['scores'].cpu().numpy()
            labels_cache = retina_out['labels'].cpu().numpy()
            
        contador_cuadros += 1
        
        p_phone, p_paper = 0.0, 0.0
        texto_depuracion = ""
        
        for box, score, label in zip(cajas_cache, scores_cache, labels_cache):
            c_id = int(label) 
            xmin, ymin, xmax, ymax = map(int, box)
            
            ancho_caja = xmax - xmin
            alto_caja = ymax - ymin
            
            if ancho_caja > 450 or alto_caja > 450:
                continue
                
            if score > 0.10:
                texto_depuracion += f"Clase: {c_id} | Certeza: {score*100:.1f}%\n"
            
            ratio_aspecto = ancho_caja / float(alto_caja) if alto_caja > 0 else 0
            
            if c_id == phone_idx and score < 0.75 and (0.75 < ratio_aspecto < 1.35):
                c_id = paper_idx

            if c_id == phone_idx and score > 0.25:
                p_phone = max(p_phone, score)
                cv2.rectangle(plotted_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2) # Rojo
                
            elif c_id == paper_idx and score > 0.10: 
                score_amplificado = min(0.98, score * 2.2)
                p_paper = max(p_paper, score_amplificado)
                cv2.rectangle(plotted_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Verde

        if texto_depuracion == "":
            texto_depuracion = "Buscando objetos..."
        debug_panel.text(texto_depuracion)

        vector_cuadro = np.concatenate(([p_phone, p_paper], pose_out))
        lista_temporal.append(vector_cuadro)
        
        if len(lista_temporal) > 15:
            lista_temporal.pop(0)
            
        if len(lista_temporal) == 15:
            matriz_secuencia = np.array(lista_temporal)
            input_tensor = torch.tensor(matriz_secuencia, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                score_lstm = lstm_net(input_tensor).item()
        else:
            score_lstm = 0.0
        
        score_final = score_lstm
        if p_phone > 0.30:
            score_final = max(score_final, p_phone)
            
        if p_paper > 0.25:
            score_final = max(score_final, p_paper + 0.45)
            
        score_final = min(1.0, float(score_final))
        
        cv2.putText(plotted_rgb, f"celular: {p_phone:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(plotted_rgb, f"papel: {p_paper:.2f}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        bar.progress(score_final)
        alto, ancho, _ = frame.shape
        
        if score_final >= threshold:
            status_text.error(f"alerta: sospecha de copia ({score_final*100:.1f}%)")
            cv2.rectangle(plotted_rgb, (0, 0), (ancho, alto), (255, 0, 0), 10) 
        else:
            status_text.success(f"estado: normal ({score_final*100:.1f}%)")
            
        VIDEO_CONTAINER.image(plotted_rgb)
        time.sleep(0.01) 
        
    cap.release()