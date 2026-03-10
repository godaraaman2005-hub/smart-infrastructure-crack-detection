#!/usr/bin/env python3
"""
Smart Infrastructure Crack Detection - Main Entry Point
Usage: python main.py --input <image_path> --output <results_dir>
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crack_detector import CrackDetectionSystem


def main():
    parser = argparse.ArgumentParser(
        description='Detect cracks in aerial infrastructure images using OpenCV'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input aerial image (JPG/PNG/TIFF)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory for results (default: results/)'
    )
    parser.add_argument(
        '--scale', '-s',
        type=float,
        default=0.5,
        help='Scale factor in cm/pixel (default: 0.5)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=50,
        help='Minimum crack area in pixels (default: 50)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=20,
        help='Minimum crack length in pixels (default: 20)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SMART INFRASTRUCTURE - CRACK DETECTION SYSTEM")
    print("Aerial Image Processing for Structural Health Monitoring")
    print("=" * 70)
    
    try:
        # Initialize detector
        detector = CrackDetectionSystem(args.input, scale_factor=args.scale)
        
        # Execute pipeline
        print("\n[1/3] Preprocessing image...")
        detector.preprocess_image()
        
        print("\n[2/3] Detecting and quantifying cracks...")
        detector.detect_and_quantify_cracks(
            min_area=args.min_area,
            min_length=args.min_length
        )
        
        print("\n[3/3] Generating reports...")
        detector.export_results(args.output)
        
        # Console summary
        report = detector.generate_report()
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Total cracks detected: {report['summary']['total_cracks']}")
        print("\nSeverity Distribution:")
        for sev, count in report['summary']['severity_distribution'].items():
            bar = "█" * count + "░" * (10 - count)
            print(f"  {sev:10}: {bar} ({count})")
        
        print(f"\nResults saved to: {os.path.abspath(args.output)}/")
        print("  - Processing stage images (01-09)")
        print("  - analysis_report.json (structured data)")
        print("  - analysis_report.txt (human-readable)")
        print("  - processing_pipeline.png (visualization)")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
