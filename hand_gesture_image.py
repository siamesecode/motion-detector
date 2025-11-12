import cv2
import mediapipe as mp
import os

class HandGestureImageDisplay:
    def __init__(self, image_paths=None):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        if image_paths is None:
            image_paths = []
            images_dir = 'images'
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_paths.append(os.path.join(images_dir, file))
        
        self.display_images = []
        self.image_names = []
        
        for img_path in image_paths:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = self.resize_image(img, 500)
                self.display_images.append(img)
                self.image_names.append(os.path.basename(img_path))
        
        if not self.display_images:
            print("Nenhuma imagem encontrada!")
        
        self.both_hands_up = False
        self.current_image_index = 0
        self.current_gesture = None
        
        self.gesture_to_image = {}
        for i, name in enumerate(self.image_names):
            name_lower = name.lower()
            if 'calabreso' in name_lower:
                self.gesture_to_image['vertical'] = i
            elif 'avril' in name_lower:
                self.gesture_to_image['inclined'] = i
            elif 'macaquinho' in name_lower or 'macaco' in name_lower:
                self.gesture_to_image['pointing_left'] = i
            elif 'fazol' in name_lower or 'faz' in name_lower:
                self.gesture_to_image['L_right'] = i
        
    def resize_image(self, img, max_size=500):
        height, width = img.shape[:2]
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        return cv2.resize(img, (new_width, new_height))
    
    def are_hands_raised(self, hand_landmarks, hand_label):
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        return middle_finger_tip.y < wrist.y - 0.1
    
    def get_hand_angle(self, hand_landmarks):
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        z_diff = middle_finger_tip.z - wrist.z
        
        if z_diff < -0.05:
            return 'inclined'
        else:
            return 'vertical'
    
    def is_pointing_up(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        index_extended = index_tip.y < index_mcp.y - 0.05
        index_up = index_tip.y < wrist.y - 0.15
        middle_closed = middle_tip.y > middle_mcp.y - 0.02
        ring_closed = ring_tip.y > ring_mcp.y - 0.02
        
        return index_extended and index_up and (middle_closed or ring_closed)
    
    def is_L_shape(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        index_extended = index_tip.y < index_mcp.y - 0.05
        thumb_extended = abs(thumb_tip.x - wrist.x) > 0.08
        thumb_horizontal = abs(thumb_tip.y - thumb_ip.y) < 0.1
        middle_closed = middle_tip.y > middle_mcp.y - 0.03
        ring_closed = ring_tip.y > wrist.y - 0.05
        
        return index_extended and thumb_extended and (middle_closed or ring_closed)
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_type = None
        hands_detected = 0
        debug_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = len(results.multi_hand_landmarks)
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                hand_label = handedness.classification[0].label
                is_pointing = self.is_pointing_up(hand_landmarks)
                is_L = self.is_L_shape(hand_landmarks)
                
                debug_info.append(f"{hand_label}: Point={is_pointing} L={is_L}")
                
                if hand_label == 'Left' and is_pointing:
                    gesture_type = 'pointing_left'
                    break
                elif hand_label == 'Right' and is_L:
                    gesture_type = 'L_right'
                    break
            
            if gesture_type is None and hands_detected >= 2:
                hand_angles = []
                hands_raised_count = 0
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    if self.are_hands_raised(hand_landmarks, hand_label):
                        hands_raised_count += 1
                        angle = self.get_hand_angle(hand_landmarks)
                        hand_angles.append(angle)
                
                if hands_raised_count >= 2:
                    if all(angle == 'vertical' for angle in hand_angles):
                        gesture_type = 'vertical'
                    elif all(angle == 'inclined' for angle in hand_angles):
                        gesture_type = 'inclined'
        
        return frame, hands_detected, gesture_type, debug_info
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return
        
        print("=" * 60)
        print("DETECTOR DE GESTOS DE MÃO")
        print("=" * 60)
        print("Gestos:")
        print("👐 DUAS MAOS RETAS → Calabreso")
        print("👐 DUAS MAOS INCLINADAS → Avril")
        print("☝️  MAO ESQUERDA APONTANDO → Macaco")
        print("🤙 MAO DIREITA L → Faz o L")
        print("\nPressione 'q' para sair")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, hands_count, gesture_type, debug_info = self.process_frame(frame)
            
            if gesture_type and gesture_type in self.gesture_to_image:
                self.current_image_index = self.gesture_to_image[gesture_type]
                self.current_gesture = gesture_type
                self.both_hands_up = True
            else:
                self.both_hands_up = False
            
            cv2.putText(processed_frame, f"Maos: {hands_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if debug_info:
                for i, info in enumerate(debug_info):
                    cv2.putText(processed_frame, info, (10, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if gesture_type:
                gesture_names = {
                    'vertical': "DUAS MAOS RETAS",
                    'inclined': "DUAS MAOS INCLINADAS", 
                    'pointing_left': "APONTANDO",
                    'L_right': "FAZENDO L"
                }
                gesture_name = gesture_names.get(gesture_type, gesture_type)
                cv2.putText(processed_frame, f"Gesto: {gesture_name}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if self.both_hands_up and self.current_gesture:
                image_name = self.image_names[self.current_image_index]
                cv2.putText(processed_frame, f"Exibindo: {image_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "Faca um gesto!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Camera - Detector de Gestos', processed_frame)
            
            if self.both_hands_up and self.current_gesture:
                current_img = self.display_images[self.current_image_index]
                cv2.imshow('Imagem - Gesto Detectado', current_img)
            else:
                if cv2.getWindowProperty('Imagem - Gesto Detectado', cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow('Imagem - Gesto Detectado')
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

def main():
    detector = HandGestureImageDisplay()
    detector.run()

if __name__ == "__main__":
    main()
