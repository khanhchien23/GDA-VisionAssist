import argparse
import os
import timeit

def main():
    """Entry point for GDA Application"""
    
    # Default checkpoint paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_SEG_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "setr_dino_best.pth")
    DEFAULT_ADAPTOR_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "adaptor_vizwiz", "adaptor.pth")
    
    parser = argparse.ArgumentParser(
        description="GDA-VisionAssist"
    )
    parser.add_argument(
        '--seg-checkpoint', 
        type=str,
        default=DEFAULT_SEG_CHECKPOINT,
        help='Segmentation decoder checkpoint (COCO-Stuff 171 classes)'
    )
    parser.add_argument(    
        '--adaptor-checkpoint', 
        type=str,
        default=DEFAULT_ADAPTOR_CHECKPOINT,
        help='Adaptor checkpoint directory (contains adaptor.pth, masked_extractor.pth, text_decoder.pth)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Print checkpoint info
    print("="*60)
    print("🔧 GDA Checkpoints:")
    print(f"   SETR: {args.seg_checkpoint}")
    print(f"   Adaptor: {args.adaptor_checkpoint}")
    print("="*60)
    
    # Import here to avoid slow startup for --help
    from src.app import GDAApplication
    
    app = GDAApplication(
        seg_checkpoint=args.seg_checkpoint,
        adaptor_checkpoint=args.adaptor_checkpoint,
        debug=args.debug
    )
    
    app.run()

if __name__ == "__main__":
    main()
