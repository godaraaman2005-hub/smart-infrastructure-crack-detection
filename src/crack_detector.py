
"""
Smart Infrastructure - Crack Detection System
Main detector class for aerial image analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os


class CrackDetectionSystem:
    """
    Complete crack detection pipeline for aerial infrastructure images.
    """
    
    def __init__(self, image_path, scale_factor=0.5):
        """
        Initialize crack detection system.
        
        Args:
            image_path: Path to aerial image
            scale_factor: cm per pixel for real-world conversion
        """
        self.image_path = image_path
        self.scale_factor = scale_factor
        
        # Load image
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.processed_images = {}
        self.crack_data = []
        self.valid_contours = []
        
        print(f"[INIT] Loaded image: {self.width}x{self.height} pixels")
    
    # =========================================================================
    # PART A: PREPROCESSING PIPELINE
    # =========================================================================
    
    def preprocess_image(self):
        """
        Stage A: Preprocess aerial images to highlight cracks.
        
        Pipeline:
        1. Grayscale conversion
        2. Gaussian blur (noise reduction)
        3. CLAHE (contrast enhancement)
        4. Adaptive thresholding
        5. Morphological operations
        6. Canny edge detection
        7. Image combination
        """
        print("[A] Starting preprocessing pipeline...")
        
        # Step 1: Grayscale Conversion
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.processed_images['01_original'] = self.original.copy()
        self.processed_images['02_grayscale'] = gray.copy()
        print("  [1/7] Grayscale conversion complete")
        
        # Step 2: Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.processed_images['03_blurred'] = blurred.copy()
        print("  [2/7] Gaussian blur applied (5x5 kernel)")
        
        # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        self.processed_images['04_enhanced'] = enhanced.copy()
        print("  [3/7] CLAHE enhancement applied")
        
        # Step 4: Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        self.processed_images['05_binary'] = binary.copy()
        print("  [4/7] Adaptive thresholding complete")
        
        # Step 5: Morphological Operations
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        self.processed_images['06_morphological'] = morph.copy()
        print("  [5/7] Morphological operations complete")
        
        # Step 6: Canny Edge Detection
        edges = cv2.Canny(enhanced, 50, 150)
        self.processed_images['07_edges'] = edges.copy()
        print("  [6/7] Canny edge detection complete")
        
        # Step 7: Combine Binary and Edges
        combined = cv2.bitwise_or(morph, edges)
        self.processed_images['08_preprocessed'] = combined.copy()
        print("  [7/7] Combined preprocessing complete")
        
        return combined
    
    # =========================================================================
    # PART B: CONTOUR DETECTION AND QUANTIFICATION
    # =========================================================================
    
    def detect_and_quantify_cracks(self, min_area=50, min_length=20):
        """
        Stage B: Detect contours, quantify dimensions, classify severity.
        """
        print("[B] Starting crack detection and quantification...")
        
        preprocessed = self.processed_images.get('08_preprocessed')
        if preprocessed is None:
            preprocessed = self.preprocess_image()
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"  Found {len(contours)} raw contours")
        
        output = self.original.copy()
        valid_contours = []
        crack_data = []
        crack_id = 0
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, False)
            
            # Filter by minimum criteria
            if area < min_area or length < min_length:
                continue
            
            crack_id += 1
            valid_contours.append(cnt)
            
            # Calculate metrics
            x, y, w, h = cv2.boundingRect(cnt)
            avg_width = (2 * area) / length if length > 0 else 0
            
            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Classify severity
            severity = self._classify_severity(area, length, avg_width)
            
            crack_info = {
                'id': crack_id,
                'pixel_metrics': {
                    'area': int(area),
                    'length': int(length),
                    'avg_width': round(avg_width, 2),
                    'bounding_box': {'x': x, 'y': y, 'w': w, 'h': h}
                },
                'real_world_metrics': {
                    'area_cm2': round(area * (self.scale_factor ** 2), 2),
                    'length_cm': round(length * self.scale_factor, 2),
                    'width_cm': round(avg_width * self.scale_factor, 2)
                },
                'location': {'centroid_x': cx, 'centroid_y': cy},
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
            crack_data.append(crack_info)
            
            # Visual annotation
            color = self._get_severity_color(severity)
            cv2.drawContours(output, [cnt], -1, color, 2)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 1)
            
            label = f"#{crack_id} {severity}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x, y-text_h-10), (x+text_w, y), color, -1)
            cv2.putText(output, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            measure_text = f"L:{crack_info['real_world_metrics']['length_cm']}cm"
            cv2.putText(output, measure_text, (x, y+h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        self.crack_data = crack_data
        self.valid_contours = valid_contours
        self.processed_images['09_detected'] = output.copy()
        
        print(f"  [✓] Validated {len(crack_data)} cracks after filtering")
        print(f"  Severity distribution: {self._get_severity_distribution()}")
        
        return crack_data, output
    
    def _classify_severity(self, area, length, avg_width):
        """
        Classify crack severity based on dimensions.
        """
        if length > 500 or avg_width > 15 or area > 5000:
            return "CRITICAL"
        elif length > 300 or avg_width > 10 or area > 2000:
            return "HIGH"
        elif length > 100 or avg_width > 5 or area > 500:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_severity_color(self, severity):
        """Return BGR color codes for severity levels."""
        colors = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 165, 255),
            "MEDIUM": (0, 255, 255),
            "LOW": (0, 255, 0)
        }
        return colors.get(severity, (128, 128, 128))
    
    def _get_severity_distribution(self):
        """Return count of cracks by severity level."""
        distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for crack in self.crack_data:
            distribution[crack['severity']] += 1
        return distribution
    
    def generate_report(self, save_path=None):
        """Generate comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        severity_dist = self._get_severity_distribution()
        
        report = {
            'metadata': {
                'project': 'Smart Infrastructure Crack Detection',
                'version': '1.0.0',
                'timestamp': timestamp,
                'image_file': self.image_path,
                'image_dimensions': {'width': self.width, 'height': self.height},
                'scale_factor': f"{self.scale_factor} cm/pixel"
            },
            'summary': {
                'total_cracks': len(self.crack_data),
                'severity_distribution': severity_dist
            },
            'cracks': self.crack_data,
            'recommendations': self._generate_recommendations(severity_dist)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[✓] JSON report saved: {save_path}")
        
        return report
    
    def _generate_recommendations(self, severity_dist):
        """Generate maintenance recommendations."""
        recommendations = []
        
        if severity_dist["CRITICAL"] > 0:
            recommendations.append(
                f"URGENT: {severity_dist['CRITICAL']} critical crack(s) detected. "
                "Immediate structural assessment required."
            )
        
        if severity_dist["HIGH"] > 0:
            recommendations.append(
                f"PRIORITY: {severity_dist['HIGH']} high-severity crack(s). "
                "Schedule repair within 30 days."
            )
        
        if severity_dist["MEDIUM"] > 0:
            recommendations.append(
                f"MONITOR: {severity_dist['MEDIUM']} medium-severity crack(s). "
                "Include in monthly inspection cycle."
            )
        
        if severity_dist["LOW"] > 0:
            recommendations.append(
                f"ROUTINE: {severity_dist['LOW']} low-severity crack(s). "
                "Standard annual maintenance sufficient."
            )
        
        if sum(severity_dist.values()) == 0:
            recommendations.append("No cracks detected. Infrastructure appears sound.")
        
        return recommendations
    
    def visualize_pipeline(self, save_dir=None):
        """Create comprehensive visualization of all processing stages."""
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle('Crack Detection Processing Pipeline - Smart Infrastructure', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        stages = [
            ('01_original', 'A. Original Aerial Image'),
            ('02_grayscale', 'A. Grayscale Conversion'),
            ('03_blurred', 'A. Gaussian Blur'),
            ('04_enhanced', 'A. CLAHE Enhancement'),
            ('05_binary', 'A. Adaptive Thresholding'),
            ('06_morphological', 'A. Morphological Operations'),
            ('07_edges', 'A. Canny Edge Detection'),
            ('08_preprocessed', 'A. Combined Preprocessing'),
            ('09_detected', 'B. Crack Detection & Classification')
        ]
        
        for idx, (key, title) in enumerate(stages):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            if key in self.processed_images:
                img = self.processed_images[key]
                if len(img.shape) == 3:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(img, cmap='gray')
                ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
            else:
                ax.text(0.5, 0.5, 'Not Processed', ha='center', va='center')
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, 'processing_pipeline.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"[✓] Pipeline visualization saved: {path}")
        
        return fig
    
    def export_results(self, output_dir="results"):
        """Export all results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[EXPORT] Saving results to: {output_dir}/")
        
        # Save all processing stages
        for name, img in self.processed_images.items():
            path = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(path, img)
        
        # Save JSON report
        json_path = os.path.join(output_dir, "analysis_report.json")
        self.generate_report(json_path)
        
        # Save text report
        text_report = self._format_text_report()
        txt_path = os.path.join(output_dir, "analysis_report.txt")
        with open(txt_path, 'w') as f:
            f.write(text_report)
        
        # Save pipeline visualization
        self.visualize_pipeline(output_dir)
        
        print(f"[✓] Exported {len(self.processed_images)} images + reports")
        return output_dir
    
    def _format_text_report(self):
        """Format human-readable text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("SMART INFRASTRUCTURE - CRACK DETECTION ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Image: {self.image_path}")
        lines.append(f"Dimensions: {self.width} x {self.height} pixels")
        lines.append(f"Scale: {self.scale_factor} cm/pixel")
        lines.append("=" * 70)
        
        lines.append(f"\nSUMMARY")
        lines.append(f"Total Cracks Detected: {len(self.crack_data)}")
        lines.append("-" * 70)
        
        dist = self._get_severity_distribution()
        for sev, count in dist.items():
            lines.append(f"  {sev:10}: {count:3d} crack(s)")
        
        lines.append("\n" + "=" * 70)
        lines.append("DETAILED CRACK ANALYSIS")
        lines.append("=" * 70)
        
        for crack in self.crack_data:
            lines.append(f"\nCrack #{crack['id']} [{crack['severity']}]")
            lines.append(f"  Location: ({crack['location']['centroid_x']}, "
                        f"{crack['location']['centroid_y']})")
            lines.append(f"  Pixel Metrics:")
            lines.append(f"    - Area:   {crack['pixel_metrics']['area']} px²")
            lines.append(f"    - Length: {crack['pixel_metrics']['length']} px")
            lines.append(f"    - Width:  {crack['pixel_metrics']['avg_width']} px (avg)")
            lines.append(f"  Real-World Metrics:")
            lines.append(f"    - Area:   {crack['real_world_metrics']['area_cm2']} cm²")
            lines.append(f"    - Length: {crack['real_world_metrics']['length_cm']} cm")
            lines.append(f"    - Width:  {crack['real_world_metrics']['width_cm']} cm")
        
        lines.append("\n" + "=" * 70)
        lines.append("MAINTENANCE RECOMMENDATIONS")
        lines.append("=" * 70)
        for rec in self._generate_recommendations(dist):
            lines.append(f"\n• {rec}")
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        return "\n".join(lines)
