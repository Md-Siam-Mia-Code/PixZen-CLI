# pixzen/cli.py
import argparse
from pathlib import Path
from pixzen.enhancer import PixZenEnhancer
from pixzen.utils import (
    REALESRGAN_MODELS,
    GFPGAN_MODELS,
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
)


def main() -> None:
    """Main entry point for the PixZen command-line interface."""
    parser = argparse.ArgumentParser(
        description="PixZen: AI-powered Image and Video Enhancement",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/Input"),
        help="Input image, video, or folder. Default: data/Input",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/Output"),
        help="Output folder. Default: data/Output",
    )
    parser.add_argument(
        "-r",
        "--realesrgan",
        type=str,
        default="v2",
        choices=[m["alias"] for m in REALESRGAN_MODELS.values()],
        help="Real-ESRGAN model version. Default: v2. Options: v1, v2, Anime, AnimeVideo",
    )
    parser.add_argument(
        "-s",
        "--outscale",
        type=float,
        default=2,
        help="The final upsampling scale of the image. Default: 2. Options: 1, 2, 3, 4",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="PixZen",
        help="Prefix of the restored image. Default: PixZen",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=400,
        help="Tile size for reducing VRAM usage. 0 for no tile. Default: 400",
    )
    parser.add_argument(
        "-g",
        "--gfpgan",
        type=str,
        default=None,
        choices=GFPGAN_MODELS.keys(),
        help="Enable GFPGAN face enhancement with a specific model version. Default: None. Options: v1, v2",
    )
    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        default="png",
        choices=["auto", "jpg", "png"],
        help="Output image extension. 'auto' uses the input format. Default: png",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        enhancer = PixZenEnhancer(args)

        # --- PRE-FLIGHT CHECK ---
        # This will download all necessary models before any processing starts.
        enhancer.prepare_models()

        if not args.input.exists():
            print(f"Error: Input path '{args.input}' does not exist.")
            return

        if args.input.is_file():
            paths = [args.input]
        else:
            paths = sorted(
                [
                    p
                    for p in args.input.glob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
                ]
            )

        if not paths:
            print(f"No supported image or video files found in '{args.input}'.")
            return

        image_paths = [p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS]
        video_paths = [p for p in paths if p.suffix.lower() in VIDEO_EXTENSIONS]

        if image_paths:
            print(f"\nFound {len(image_paths)} image(s) to process...")
            for path in image_paths:
                enhancer.enhance_image(path)
            print("-" * 50)

        if video_paths:
            print(f"\nFound {len(video_paths)} video(s) to process...")
            for path in video_paths:
                enhancer.enhance_video(path)

        print("\nAll tasks completed.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
