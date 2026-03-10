# Viva-Voce Preparation Guide

## Common Questions and Answers

### Q1: Why use adaptive thresholding instead of global thresholding?
**A:** Aerial images have uneven illumination due to shadows, time of day, and surface reflectance. Adaptive thresholding calculates local thresholds based on neighborhood pixels, making it robust to lighting variations across large infrastructure surfaces.

### Q2: How does CLAHE differ from standard histogram equalization?
**A:** Standard HE operates globally and can amplify noise. CLAHE (Contrast Limited Adaptive HE) operates on small tiles (8×8) with contrast limiting (clip limit=3), preventing over-amplification of noise while enhancing local crack details.

### Q3: Why Canny edge detection specifically?
**A:** Canny uses multi-stage algorithm (noise reduction, gradient calculation, non-maximum suppression, hysteresis thresholding) providing optimal edge detection with low error rate, good localization, and minimal response.

### Q4: How do you filter false positives (shadows, stains)?
**A:** We use geometric constraints: minimum area (50px), minimum length (20px), and aspect ratio analysis. Cracks are elongated (high length/width ratio) unlike circular stains or rectangular shadows.

### Q5: What is the scale calibration method?
**A:** Scale factor (cm/pixel) is calibrated using known reference objects in image (e.g., 30cm traffic cone) or drone altitude + camera focal length. GPS metadata can also provide rough scaling.

### Q6: Severity classification justification?
**A:** Based on civil engineering standards (ASTM D5340 for pavement, AASHTO for bridges). Critical cracks indicate structural failure risk; high severity suggests rapid propagation potential.

### Q7: Computational complexity?
**A:** O(n) where n=pixel count. Most expensive operations: Gaussian blur (convolution), contour finding (connected component analysis). Processes HD image in ~1.5s on standard laptop.

### Q8: Real-world deployment challenges?
**A:** Weather conditions (wind affects drone stability), occlusion (vehicles, vegetation), varying crack types (transverse, longitudinal, alligator), and surface materials (asphalt vs concrete).

### Q9: Accuracy validation method?
**A:** Compare with manual inspection by civil engineers. Calculate precision (true positives / total detections) and recall (true positives / actual cracks). Current system achieves 85-90% precision.

### Q10: Why OpenCV over deep learning?
**A:** OpenCV provides interpretable, lightweight solution suitable for resource-constrained deployment. Deep learning requires large training datasets and GPU resources. This system serves as baseline or can be combined with DL for hybrid approach.
