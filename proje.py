"""
=============================================================================
AKILLİ ŞERİT TAKİP VE DİREKSİYON ASİSTAN SİSTEMİ
Bilgisayar Görü Tabanlı Otonom Sürüş Teknolojisi
=============================================================================

Proje: Gerçek zamanlı şerit tespit ve direksiyon asistan sistemi
Teknolojiler: OpenCV, NumPy, Python
Özellikler: 
- Gelişmiş şerit tespiti
- Direksiyon simülasyonu
- Şerit değişimi tespiti
- Yörünge tahmini
- Gerçek zamanlı analiz

Geliştirici: Melih DURMAZ, Muhammad Ali DUYAR
Tarih: 2025
Versiyon: 13.0
=============================================================================
"""

import cv2
import numpy as np
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging

from ultralytics import YOLO

# Logging yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LaneDetectionConfig:
    """Şerit tespit parametreleri"""
    # Görüntü işleme parametreleri
    GAUSSIAN_KERNEL_SIZE: int = 5
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150
    
    # Hough Transform parametreleri
    HOUGH_RHO: int = 2
    HOUGH_THETA: float = np.pi/180
    HOUGH_THRESHOLD: int = 100
    HOUGH_MIN_LINE_LENGTH: int = 40
    HOUGH_MAX_LINE_GAP: int = 20
    
    # Şerit filtreleme parametreleri
    MIN_SLOPE_THRESHOLD: float = 0.5
    MAX_SLOPE_THRESHOLD: float = 2.0
    
    # ROI parametreleri
    ROI_TOP_RATIO: float = 0.6
    ROI_LEFT_RATIO: float = 0.45
    ROI_RIGHT_RATIO: float = 0.55
    
    # Yumuşatma parametreleri
    SMOOTHING_WEIGHT: float = 0.7
    STEERING_SMOOTHING_WEIGHT: float = 0.6
    
    # Şerit değişimi parametreleri
    LANE_CHANGE_VARIANCE_THRESHOLD: float = 500
    TRANSITION_DURATION_FRAMES: int = 45

@dataclass
class SteeringData:
    """Direksiyon verileri"""
    angle: float
    deviation_pixels: int
    deviation_cm: float
    lane_center: int
    confidence: float

@dataclass
class LaneData:
    """Şerit verileri"""
    left_line: List[int]
    right_line: List[int]
    center_bottom: int
    center_top: int
    width: int
    confidence: float

def estimate_distance(bbox_width, bbox_height):
    # For simplicity, assume the distance is inversely proportional to the box size
    # This is a basic estimation, you may use camera calibration for more accuracy
        focal_length = 1000  # Example focal length, modify based on camera setup
        known_width = 2.0  # Approximate width of the car (in meters)
        distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
        return distance

class AdvancedLaneDetectionSystem:
    """Gelişmiş Şerit Tespit ve Direksiyon Asistan Sistemi"""
    
    def __init__(self, config: LaneDetectionConfig = None):
        """Sistem başlatma"""
        self.config = config or LaneDetectionConfig()
        
        # Geçmiş verileri
        self.lane_history = {
            'left_lines': deque(maxlen=10),
            'right_lines': deque(maxlen=10),
            'lane_center_history': deque(maxlen=15),
            'steering_history': deque(maxlen=8),
            'green_area_center': deque(maxlen=5)
        }
        
        # Sistem durumu
        self.system_state = {
            'transition_frames': 0,
            'is_changing_lanes': False,
            'current_steering_angle': 0.0,
            'last_detection_time': time.time(),
            'frame_count': 0,
            'fps': 0.0
        }
        
        # Performans metrikleri
        self.performance_metrics = {
            'processing_times': deque(maxlen=30),
            'detection_confidence': deque(maxlen=30),
            'steering_stability': deque(maxlen=30)
        }
        
        logger.info("Akıllı Şerit Tespit Sistemi başlatıldı")
    
    def preprocess_image(self, image: np.ndarray, adaptive: bool = False) -> np.ndarray:
        """
        Gelişmiş görüntü ön işleme
        
        Args:
            image: Giriş görüntüsü
            adaptive: Adaptif işleme kullan
            
        Returns:
            İşlenmiş kenar görüntüsü
        """
        try:
            # Gürültü azaltma
            kernel_size = 7 if adaptive else self.config.GAUSSIAN_KERNEL_SIZE
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            # Renk uzayı dönüşümü
            gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            
            # Histogram eşitleme (kontrast artırma)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Adaptif kenar tespiti
            if adaptive:
                # Şerit değişimi sırasında daha hassas
                edges = cv2.Canny(enhanced, 30, 120)
            else:
                edges = cv2.Canny(enhanced, 
                                self.config.CANNY_LOW_THRESHOLD, 
                                self.config.CANNY_HIGH_THRESHOLD)
            
            return edges
            
        except Exception as e:
            logger.error(f"Görüntü ön işleme hatası: {e}")
            return np.zeros_like(image[:,:,0])
    
    def get_adaptive_roi_vertices(self, height: int, width: int) -> np.ndarray:
        """
        Adaptif ilgi alanı hesaplama
        
        Args:
            height: Görüntü yüksekliği
            width: Görüntü genişliği
            
        Returns:
            ROI köşe noktaları
        """
        lane_changing = self.system_state['is_changing_lanes']
        
        if lane_changing:
            # Şerit değişimi sırasında daha geniş ROI
            left_ratio = 0.4
            right_ratio = 0.6
            top_ratio = 0.55
        else:
            left_ratio = self.config.ROI_LEFT_RATIO
            right_ratio = self.config.ROI_RIGHT_RATIO
            top_ratio = self.config.ROI_TOP_RATIO
        
        return np.array([[
            (0, height),
            (width * left_ratio, height * top_ratio),
            (width * right_ratio, height * top_ratio),
            (width, height)
        ]], dtype=np.int32)
    
    def apply_roi_mask(self, image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """ROI maskesi uygula"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)
    
    def detect_lane_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Şerit çizgilerini tespit et
        
        Args:
            masked_edges: Maskelenmiş kenar görüntüsü
            
        Returns:
            Tespit edilen çizgiler
        """
        try:
            # Adaptif Hough parametreleri
            if self.system_state['is_changing_lanes']:
                threshold = 80
                min_line_length = 30
                max_line_gap = 30
            else:
                threshold = self.config.HOUGH_THRESHOLD
                min_line_length = self.config.HOUGH_MIN_LINE_LENGTH
                max_line_gap = self.config.HOUGH_MAX_LINE_GAP
            
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=self.config.HOUGH_RHO,
                theta=self.config.HOUGH_THETA,
                threshold=threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
            
            return lines
            
        except Exception as e:
            logger.error(f"Çizgi tespit hatası: {e}")
            return None
    
    def classify_and_filter_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """
        Çizgileri sol/sağ olarak sınıflandır ve filtrele
        
        Args:
            lines: Tespit edilen çizgiler
            
        Returns:
            Sol ve sağ çizgi koordinatları
        """
        left_line_x, left_line_y = [], []
        right_line_x, right_line_y = [], []
        
        lane_changing = self.system_state['is_changing_lanes']
        min_slope = 0.3 if lane_changing else self.config.MIN_SLOPE_THRESHOLD
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Eğim hesapla
            if (x2 - x1) == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Eğim filtresi
            if abs(slope) < min_slope or abs(slope) > self.config.MAX_SLOPE_THRESHOLD:
                continue
            
            # Çizgi uzunluğu kontrolü
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if line_length < 20:
                continue
            
            # Sol/sağ sınıflandırma
            if slope <= 0:  # Sol şerit (negatif eğim)
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # Sağ şerit (pozitif eğim)
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
        
        return (left_line_x, left_line_y), (right_line_x, right_line_y)
    
    def fit_lane_polynomials(self, left_coords: Tuple, right_coords: Tuple, 
                           height: int) -> Tuple[List[int], List[int]]:
        """
        Şerit polinomlarını uydur
        
        Args:
            left_coords: Sol şerit koordinatları
            right_coords: Sağ şerit koordinatları
            height: Görüntü yüksekliği
            
        Returns:
            Sol ve sağ şerit çizgi koordinatları
        """
        min_y = int(height * self.config.ROI_TOP_RATIO)
        max_y = height
        
        left_line_x, left_line_y = left_coords
        right_line_x, right_line_y = right_coords
        
        # Sol şerit
        if left_line_x and left_line_y and len(left_line_x) >= 2:
            try:
                poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))
            except:
                left_x_start, left_x_end = 0, 0
        else:
            left_x_start, left_x_end = 0, 0
        
        # Sağ şerit
        if right_line_x and right_line_y and len(right_line_x) >= 2:
            try:
                poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))
            except:
                right_x_start, right_x_end = 0, 0
        else:
            right_x_start, right_x_end = 0, 0
        
        return ([left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y])
    
    def smooth_line_coordinates(self, new_coords: List[int], 
                              history: deque) -> List[int]:
        """Çizgi koordinatlarını yumuşat"""
        if not history:
            history.append(new_coords)
            return new_coords
        
        last_coords = history[-1]
        smoothed = []
        weight = self.config.SMOOTHING_WEIGHT
        
        for i in range(len(new_coords)):
            if new_coords[i] != 0 and last_coords[i] != 0:
                smoothed_val = int(weight * new_coords[i] + (1 - weight) * last_coords[i])
                smoothed.append(smoothed_val)
            elif new_coords[i] != 0:
                smoothed.append(new_coords[i])
            else:
                smoothed.append(last_coords[i])
        
        history.append(smoothed)
        return smoothed
    
    def calculate_lane_data(self, left_line: List[int], 
                          right_line: List[int]) -> LaneData:
        """
        Şerit verilerini hesapla
        
        Args:
            left_line: Sol şerit koordinatları
            right_line: Sağ şerit koordinatları
            
        Returns:
            Şerit verileri
        """
        # Güven skoru hesapla
        confidence = 0.0
        if left_line[0] != 0:
            confidence += 0.5
        if right_line[0] != 0:
            confidence += 0.5
        
        # Merkez hesaplama
        if left_line[0] != 0 and right_line[0] != 0:
            center_bottom = (left_line[0] + right_line[0]) // 2
            center_top = (left_line[2] + right_line[2]) // 2
            width = abs(right_line[0] - left_line[0])
        elif left_line[0] != 0:
            # Sol şerit mevcut, sağı tahmin et
            estimated_right_bottom = left_line[0] + 600
            estimated_right_top = left_line[2] + 600
            center_bottom = (left_line[0] + estimated_right_bottom) // 2
            center_top = (left_line[2] + estimated_right_top) // 2
            width = 600
            confidence *= 0.7  # Tahmin olduğu için güveni azalt
        elif right_line[0] != 0:
            # Sağ şerit mevcut, solu tahmin et
            estimated_left_bottom = right_line[0] - 600
            estimated_left_top = right_line[2] - 600
            center_bottom = (estimated_left_bottom + right_line[0]) // 2
            center_top = (estimated_left_top + right_line[2]) // 2
            width = 600
            confidence *= 0.7
        else:
            center_bottom = center_top = width = 0
            confidence = 0.0
        
        return LaneData(
            left_line=left_line,
            right_line=right_line,
            center_bottom=center_bottom,
            center_top=center_top,
            width=width,
            confidence=confidence
        )
    
    def calculate_steering_data(self, lane_data: LaneData, 
                              image_center: int, image_height: int) -> SteeringData:
        """
        Direksiyon verilerini hesapla
        
        Args:
            lane_data: Şerit verileri
            image_center: Görüntü merkezi
            image_height: Görüntü yüksekliği
            
        Returns:
            Direksiyon verileri
        """
        if lane_data.center_bottom == 0:
            return SteeringData(0.0, 0, 0.0, 0, 0.0)
        
        # Ağırlıklı merkez hesaplama (alt kısım daha önemli)
        weighted_center = (lane_data.center_bottom * 0.7 + lane_data.center_top * 0.3)
        
        # Sapma hesaplama
        deviation_pixels = int(weighted_center - image_center)
        deviation_cm = deviation_pixels * 0.3  # Piksel to cm dönüşümü
        
        # Direksiyon açısı hesaplama
        max_deviation = image_center
        normalized_deviation = deviation_pixels / max_deviation
        steering_angle = normalized_deviation * 45  # Maksimum 45 derece
        
        # Sınırlama
        steering_angle = max(-45, min(45, steering_angle))
        
        return SteeringData(
            angle=steering_angle,
            deviation_pixels=deviation_pixels,
            deviation_cm=deviation_cm,
            lane_center=int(weighted_center),
            confidence=lane_data.confidence
        )
    
    def smooth_steering_angle(self, new_angle: float) -> float:
        """Direksiyon açısını yumuşat"""
        history = self.lane_history['steering_history']
        
        if not history:
            history.append(new_angle)
            return new_angle
        
        last_angle = history[-1]
        weight = self.config.STEERING_SMOOTHING_WEIGHT
        smoothed_angle = weight * new_angle + (1 - weight) * last_angle
        
        history.append(smoothed_angle)
        return smoothed_angle
    
    def detect_lane_change(self, lane_data: LaneData) -> bool:
        """Şerit değişimi tespit et"""
        center_history = self.lane_history['lane_center_history']
        
        if len(center_history) < 5 or lane_data.center_bottom == 0:
            return False
        
        center_history.append(lane_data.center_bottom)
        
        # Son 5 frame'deki varyans kontrolü
        recent_centers = list(center_history)[-5:]
        center_variance = np.var(recent_centers)
        
        return center_variance > self.config.LANE_CHANGE_VARIANCE_THRESHOLD
    
    def update_system_state(self, lane_change_detected: bool):
        """Sistem durumunu güncelle"""
        if lane_change_detected and not self.system_state['is_changing_lanes']:
            self.system_state['is_changing_lanes'] = True
            self.system_state['transition_frames'] = 0
            logger.info("Şerit değişimi tespit edildi")
        
        if self.system_state['is_changing_lanes']:
            self.system_state['transition_frames'] += 1
            
            if self.system_state['transition_frames'] > self.config.TRANSITION_DURATION_FRAMES:
                self.system_state['is_changing_lanes'] = False
                self.system_state['transition_frames'] = 0
                logger.info("Şerit değişimi tamamlandı")
    
    def draw_lane_overlay(self, image: np.ndarray, lane_data: LaneData) -> np.ndarray:
        """Şerit overlay'ini çiz"""
        if all(coord == 0 for coord in lane_data.left_line + lane_data.right_line):
            return image
        
        overlay = image.copy()
        
        # Yeşil alan rengi (şerit değişimine göre)
        if self.system_state['is_changing_lanes']:
            # Geçiş efekti
            progress = min(self.system_state['transition_frames'] / 30.0, 1.0)
            green = np.array([0, 255, 0])
            blue = np.array([255, 100, 0])
            color = green + (blue - green) * progress
            alpha = 0.3 + 0.2 * math.sin(progress * math.pi * 4)
        else:
            color = [0, 255, 0]  # Yeşil
            alpha = 0.3
        
        # Polygon çiz
        points = np.array([
            [lane_data.left_line[0], lane_data.left_line[1]],
            [lane_data.left_line[2], lane_data.left_line[3]],
            [lane_data.right_line[2], lane_data.right_line[3]],
            [lane_data.right_line[0], lane_data.right_line[1]]
        ], dtype=np.int32)
        
        cv2.fillPoly(overlay, [points], color)
        
        # Şerit çizgileri
        cv2.line(overlay, 
                (lane_data.left_line[0], lane_data.left_line[1]),
                (lane_data.left_line[2], lane_data.left_line[3]),
                (0, 255, 255), 4)
        cv2.line(overlay,
                (lane_data.right_line[0], lane_data.right_line[1]),
                (lane_data.right_line[2], lane_data.right_line[3]),
                (0, 255, 255), 4)
        
        # Merkez çizgisi (şerit değişimi sırasında)
        if self.system_state['is_changing_lanes']:
            cv2.line(overlay,
                    (lane_data.center_bottom, lane_data.left_line[1]),
                    (lane_data.center_top, lane_data.left_line[3]),
                    (255, 255, 0), 3)
        
        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    def draw_steering_visualization(self, image: np.ndarray, 
                                  steering_data: SteeringData,
                                  lane_data: LaneData,
                                  image_center: int) -> np.ndarray:
        """Direksiyon görselleştirmesi"""
        height, width = image.shape[:2]
        
        # 1. Direksiyon simidi
        self._draw_steering_wheel(image, steering_data.angle)
        
        # 2. Üst panel gösterge
        self._draw_steering_indicator(image, steering_data.angle, width)
        
        # 3. Merkez çizgisi analizi
        self._draw_center_line_analysis(image, lane_data, image_center, height)
        
        # 4. Yörünge tahmini
        self._draw_trajectory_prediction(image, steering_data, lane_data, width, height)
        
        return image
    
    def _draw_steering_wheel(self, image: np.ndarray, steering_angle: float):
        """Direksiyon simidi çiz"""
        position = (100, 100)
        radius = 60
        center_x, center_y = position
        
        # Renk belirleme
        abs_angle = abs(steering_angle)
        if abs_angle < 5:
            wheel_color = (0, 255, 0)  # Yeşil
        elif abs_angle < 15:
            wheel_color = (0, 255, 255)  # Sarı
        else:
            wheel_color = (0, 0, 255)  # Kırmızı
        
        # Direksiyon çemberi
        cv2.circle(image, (center_x, center_y), radius, wheel_color, 3)
        cv2.circle(image, (center_x, center_y), radius-10, (50, 50, 50), 2)
        
        # Direksiyon kolu
        angle_rad = math.radians(steering_angle)
        end_x = int(center_x + (radius - 15) * math.sin(angle_rad))
        end_y = int(center_y - (radius - 15) * math.cos(angle_rad))
        
        cv2.line(image, (center_x, center_y), (end_x, end_y), (255, 255, 255), 4)
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Bilgi metni
        cv2.putText(image, f"Steering: {steering_angle:.1f}°",
                   (center_x - 50, center_y + radius + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, wheel_color, 2)
    
    def _draw_steering_indicator(self, image: np.ndarray, 
                               steering_angle: float, width: int):
        """Üst panel direksiyon göstergesi"""
        bar_width = 300
        bar_height = 20
        bar_x = (width - bar_width) // 2
        bar_y = 30
        
        # Arka plan
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Merkez çizgisi
        center_x = bar_x + bar_width // 2
        cv2.line(image, (center_x, bar_y), (center_x, bar_y + bar_height), 
                (255, 255, 255), 2)
        
        # Direksiyon pozisyonu
        steering_pos = int(center_x + (steering_angle / 45) * (bar_width // 2))
        steering_pos = max(bar_x, min(bar_x + bar_width, steering_pos))
        
        # Renk belirleme
        abs_angle = abs(steering_angle)
        if abs_angle < 5:
            color = (0, 255, 0)
        elif abs_angle < 15:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.circle(image, (steering_pos, bar_y + bar_height // 2), 8, color, -1)
        
        # Yön okları
        if steering_angle > 5:
            cv2.putText(image, "→", (bar_x + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif steering_angle < -5:
            cv2.putText(image, "←", (bar_x - 30, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    def _draw_center_line_analysis(self, image: np.ndarray, lane_data: LaneData,
                                 image_center: int, height: int):
        """Merkez çizgisi analizi"""
        if lane_data.center_bottom == 0:
            return
        
        # Şerit merkez çizgisi (sarı)
        cv2.line(image,
                (lane_data.center_bottom, height - 50),
                (lane_data.center_top, int(height * 0.6)),
                (255, 255, 0), 3)
        
        # Araç merkez çizgisi (beyaz)
        cv2.line(image,
                (image_center, height),
                (image_center, int(height * 0.6)),
                (255, 255, 255), 2)
        
        # Merkez noktaları
        cv2.circle(image, (lane_data.center_bottom, height - 50), 8, (0, 255, 255), -1)
        cv2.circle(image, (lane_data.center_top, int(height * 0.6)), 8, (0, 255, 255), -1)
        
        # Sapma oku
        deviation = lane_data.center_bottom - image_center
        if abs(deviation) > 10:
            arrow_start = (image_center, height - 100)
            arrow_end = (lane_data.center_bottom, height - 100)
            cv2.arrowedLine(image, arrow_start, arrow_end, (0, 0, 255), 3)
    
    def _draw_trajectory_prediction(self, image: np.ndarray, 
                                  steering_data: SteeringData,
                                  lane_data: LaneData,
                                  width: int, height: int):
        """Yörünge tahmini çiz"""
        if lane_data.center_bottom == 0 or steering_data.angle == 0:
            return
        
        # Başlangıç ve hedef noktaları
        start_x = width // 2
        start_y = height - 50
        target_x = lane_data.center_bottom
        target_y = height - 50
        
        # Bezier eğrisi noktaları
        trajectory_points = []
        steps = 20
        
        for i in range(steps + 1):
            t = i / steps
            control_x = (start_x + target_x) // 2
            control_y = start_y - 100
            
            
            # Quadratic Bezier eğrisi
            x = int((1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * target_x)
            y = int((1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * target_y)
            
            if 0 <= x < width and 0 <= y < height:
                trajectory_points.append((x, y))
        
        # Yörüngeyi gradyan renkle çiz
        if len(trajectory_points) > 1:
            for i in range(len(trajectory_points) - 1):
                alpha = i / len(trajectory_points)
                color_intensity = int(255 * (1 - alpha))
                cv2.line(image, trajectory_points[i], trajectory_points[i + 1],
                        (255, color_intensity, 255), 3)
    
    def draw_information_panel(self, image: np.ndarray, 
                             steering_data: SteeringData,
                             lane_data: LaneData,
                             fps: float) -> np.ndarray:
        """Bilgi paneli çiz"""
        panel_height = 200
        panel_width = 350
        panel_x = image.shape[1] - panel_width - 20
        panel_y = 20
        
        # Panel arka planı
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Panel çerçevesi
        cv2.rectangle(image, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        # Başlık
        cv2.putText(image, "LANE ASSIST SYSTEM", 
                   (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = 50
        line_height = 25
        
        # Sistem durumu
        status_color = (0, 255, 0) if not self.system_state['is_changing_lanes'] else (0, 255, 255)
        status_text = "NORMAL" if not self.system_state['is_changing_lanes'] else "LANE CHANGING"
        cv2.putText(image, f"Status: {status_text}",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        y_offset += line_height
        
        # Direksiyon bilgileri
        cv2.putText(image, f"Steering Angle: {steering_data.angle:.1f}°",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += line_height
        
        # Sapma bilgileri
        cv2.putText(image, f"Deviation: {steering_data.deviation_cm:.1f} cm",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += line_height
        
        # Şerit bilgileri
        cv2.putText(image, f"Lane Width: {lane_data.width} px",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += line_height
        
        # Güven skoru
        confidence_color = (0, 255, 0) if lane_data.confidence > 0.8 else (0, 255, 255) if lane_data.confidence > 0.5 else (0, 0, 255)
        cv2.putText(image, f"Confidence: {lane_data.confidence:.2f}",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
        
        y_offset += line_height
        
        # FPS
        cv2.putText(image, f"FPS: {fps:.1f}",
                   (panel_x + 10, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Güven göstergesi çubuğu
        bar_x = panel_x + 10
        bar_y = panel_y + panel_height - 30
        bar_width = panel_width - 20
        bar_height = 15
        
        # Arka plan
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Güven çubuğu
        confidence_width = int(bar_width * lane_data.confidence)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height),
                     confidence_color, -1)
        
        return image
    
    def draw_warning_alerts(self, image: np.ndarray, 
                          steering_data: SteeringData,
                          lane_data: LaneData) -> np.ndarray:
        """Uyarı mesajları"""
        warnings = []
        
        # Düşük güven uyarısı
        if lane_data.confidence < 0.3:
            warnings.append(("WARNING: LOW DETECTION CONFIDENCE", (0, 0, 255)))
        
        # Keskin dönüş uyarısı
        if abs(steering_data.angle) > 30:
            warnings.append(("SHARP TURN DETECTED", (0, 165, 255)))
        
        # Şerit sapma uyarısı
        if abs(steering_data.deviation_cm) > 50:
            warnings.append(("LANE DEPARTURE WARNING", (0, 0, 255)))
        
        # Şerit değişimi uyarısı
        if self.system_state['is_changing_lanes']:
            warnings.append(("LANE CHANGE IN PROGRESS", (0, 255, 255)))
        
        # Uyarıları çiz
        for i, (warning_text, color) in enumerate(warnings):
            y_pos = 200 + i * 30
            
            # Yanıp sönen efekt
            if int(time.time() * 3) % 2:
                cv2.putText(image, warning_text, (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def calculate_performance_metrics(self, processing_time: float,
                                    lane_data: LaneData,
                                    steering_data: SteeringData):
        """Performans metriklerini hesapla"""
        # İşlem süresi
        self.performance_metrics['processing_times'].append(processing_time)
        
        # Tespit güveni
        self.performance_metrics['detection_confidence'].append(lane_data.confidence)
        
        # Direksiyon kararlılığı
        if len(self.lane_history['steering_history']) > 1:
            steering_variance = np.var(list(self.lane_history['steering_history']))
            stability = 1.0 / (1.0 + steering_variance)  # Düşük varyans = yüksek kararlılık
            self.performance_metrics['steering_stability'].append(stability)
    
    def get_performance_report(self) -> Dict:
        """Performans raporu"""
        if not self.performance_metrics['processing_times']:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.performance_metrics['processing_times']),
            'avg_fps': 1.0 / np.mean(self.performance_metrics['processing_times']),
            'avg_confidence': np.mean(self.performance_metrics['detection_confidence']),
            'avg_stability': np.mean(self.performance_metrics['steering_stability']) if self.performance_metrics['steering_stability'] else 0,
            'total_frames': self.system_state['frame_count']
        }
    
    
    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Ana frame işleme fonksiyonu
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            İşlenmiş görüntü
        """
        start_time = time.time()
        model = YOLO('weights/yolov8n.pt')
        
        try:
            height, width = image.shape[:2]
            image_center = width // 2
            
            # 1. Görüntü ön işleme
            adaptive_processing = self.system_state['is_changing_lanes']
            edges = self.preprocess_image(image, adaptive_processing)
            
            # 2. ROI uygula
            roi_vertices = self.get_adaptive_roi_vertices(height, width)
            masked_edges = self.apply_roi_mask(edges, roi_vertices)
            
            # 3. Şerit çizgilerini tespit et
            lines = self.detect_lane_lines(masked_edges)
            
            if lines is None:
                return self._draw_no_detection_overlay(image)
            
            # 4. Çizgileri sınıflandır ve filtrele
            left_coords, right_coords = self.classify_and_filter_lines(lines)
            
            # 5. Polinom uydur
            raw_left, raw_right = self.fit_lane_polynomials(left_coords, right_coords, height)
            
            # 6. Koordinatları yumuşat
            smoothed_left = self.smooth_line_coordinates(raw_left, self.lane_history['left_lines'])
            smoothed_right = self.smooth_line_coordinates(raw_right, self.lane_history['right_lines'])
            
            # 7. Şerit verilerini hesapla
            lane_data = self.calculate_lane_data(smoothed_left, smoothed_right)
            
            # 8. Direksiyon verilerini hesapla
            steering_data = self.calculate_steering_data(lane_data, image_center, height)
            
            # 9. Direksiyon açısını yumuşat
            smooth_steering_angle = self.smooth_steering_angle(steering_data.angle)
            steering_data.angle = smooth_steering_angle
            self.system_state['current_steering_angle'] = smooth_steering_angle
            
            # 10. Şerit değişimi tespit et
            lane_change_detected = self.detect_lane_change(lane_data)
            self.update_system_state(lane_change_detected)
            
            # 11. Görselleştirme
            result_image = image.copy()
            
            # Şerit overlay'i
            result_image = self.draw_lane_overlay(result_image, lane_data)
            results = model(result_image)  # model'i global olarak tanımla
    
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    if model.names[cls] == 'car' and conf >= 0.5:
                        label = f'{model.names[cls]} {conf:.2f}'
                        
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(result_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        distance = estimate_distance(bbox_width, bbox_height)
                        
                        distance_label = f'Distance: {distance:.2f}m'
                        cv2.putText(result_image, distance_label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
            # Direksiyon görselleştirmesi
            result_image = self.draw_steering_visualization(result_image, steering_data, 
                                                                lane_data, image_center)
            
            # FPS hesaplama
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.system_state['fps'] = fps
            
            # Bilgi paneli
            result_image = self.draw_information_panel(result_image, steering_data, 
                                                     lane_data, fps)
            
            # Uyarı mesajları
            result_image = self.draw_warning_alerts(result_image, steering_data, lane_data)
            
            # Performans metrikleri
            self.calculate_performance_metrics(processing_time, lane_data, steering_data)
            
            # Frame sayacı
            self.system_state['frame_count'] += 1
            
            return result_image
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
            return self._draw_error_overlay(image, str(e))
    
    def _draw_no_detection_overlay(self, image: np.ndarray) -> np.ndarray:
        """Tespit yok overlay'i"""
        overlay = image.copy()
        cv2.putText(overlay, "NO LANE DETECTED", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return overlay
    
    def _draw_error_overlay(self, image: np.ndarray, error_msg: str) -> np.ndarray:
        """Hata overlay'i"""
        overlay = image.copy()
        cv2.putText(overlay, f"ERROR: {error_msg[:30]}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return overlay


# Ana uygulama sınıfı
class LaneAssistApplication:
    """Şerit Asistan Uygulaması"""
    
    def __init__(self, video_source=0):
        """
        Uygulama başlatma
        
        Args:
            video_source: Video kaynağı (0=webcam, dosya yolu=video dosyası)
        """
        self.video_source = video_source
        self.config = LaneDetectionConfig()
        self.detection_system = AdvancedLaneDetectionSystem(self.config)
        self.cap = None
        self.is_running = False
        
        logger.info(f"Şerit Asistan Uygulaması başlatıldı - Kaynak: {video_source}")
    
    def initialize_video_capture(self) -> bool:
        """Video yakalama başlat"""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                logger.error("Video kaynağı açılamadı")
                return False
            
            # Video özelliklerini ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Video yakalama başlatıldı")
            return True
            
        except Exception as e:
            logger.error(f"Video başlatma hatası: {e}")
            return False
    
    def run(self):
        """Ana uygulama döngüsü"""
        if not self.initialize_video_capture():
            return
        
        self.is_running = True
        logger.info("Uygulama çalışıyor... (Çıkmak için 'q' tuşuna basın)")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Frame okunamadı")
                    break
                
                # Frame işle
                processed_frame = self.detection_system.process_frame(frame)
                
                # Göster
                cv2.imshow('Akıllı Şerit Takip ve Direksiyon Asistan Sistemi', processed_frame)
                
                # Klavye kontrolü
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Sistem sıfırlama
                    self.detection_system = AdvancedLaneDetectionSystem(self.config)
                    logger.info("Sistem sıfırlandı")
                elif key == ord('p'):
                    # Performans raporu
                    report = self.detection_system.get_performance_report()
                    logger.info(f"Performans Raporu: {report}")
                
        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından durduruldu")
        except Exception as e:
            logger.error(f"Uygulama hatası: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Temizlik işlemleri"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Final performans raporu
        report = self.detection_system.get_performance_report()
        logger.info(f"Final Performans Raporu: {report}")
        logger.info("Uygulama kapatıldı")




# Ana çalıştırma
if __name__ == "__main__":
    app = LaneAssistApplication("proje_video.mp4")
    app.run()
