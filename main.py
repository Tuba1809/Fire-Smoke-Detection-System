import cv2
import argparse
import os

from src.preprocessing import (
    resize_image,
    apply_blur,
    convert_to_grayscale,
    convert_to_hsv
)

from src.segmentation import (
    create_fire_mask,
    create_smoke_mask
)

from src.morphology import clean_mask
from src.detection import find_regions
from src.video_detection import process_video

from src.classification import (
    extract_features,
    classify_fire,
    classify_smoke,
    determine_final_status
)


def print_header():
    print("\n" + "=" * 60)
    print("          FIRE & SMOKE DETECTION SYSTEM")
    print("=" * 60)


def print_final_result(
    fire_count,
    smoke_count,
    final_status,
    output_path
):
    print("\n" + "-" * 60)
    print("                    FINAL RESULT")
    print("-" * 60)

    print(f"Fire regions detected  : {fire_count}")
    print(f"Smoke regions detected : {smoke_count}")
    print(f"Detection status       : {final_status}")

    print("-" * 60)
    print(f"Output saved           : {output_path}")
    print("-" * 60)


def main():

    # ==================================================
    # 1. COMMAND-LINE ARGUMENTS
    # ==================================================

    parser = argparse.ArgumentParser(
        description="Fire and Smoke Detection using Computer Vision"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input image or video"
    )

    parser.add_argument(
        "--type",
        choices=["image", "video"],
        default="image",
        help="Input type: image or video"
    )

    parser.add_argument(
        "--output",
        default="results/images/detection_result.jpg",
        help="Path to save output image"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detection result"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed region analysis"
    )

    args = parser.parse_args()
    if args.type == "video":

        output_path = args.output

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        success = process_video(
            args.input,
            output_path,
            show=args.show
        )

        if success:
            print("\nVideo processing completed successfully.\n")

        return

    image_path = args.input
    output_path = args.output


    # ==================================================
    # 2. START
    # ==================================================

    print_header()

    print(f"\nInput image : {image_path}")


    # ==================================================
    # 3. LOAD IMAGE
    # ==================================================

    image = cv2.imread(image_path)

    if image is None:

        print("\nERROR: Could not load input image.")
        print(f"Check the file path: {image_path}")

        return


    print("Image loading          : COMPLETE")


    # ==================================================
    # 4. PREPROCESSING
    # ==================================================

    resized = resize_image(image)

    blurred = apply_blur(resized)

    grayscale = convert_to_grayscale(blurred)

    hsv = convert_to_hsv(blurred)

    print("Preprocessing           : COMPLETE")


    # ==================================================
    # 5. SEGMENTATION
    # ==================================================

    fire_mask = create_fire_mask(hsv)

    smoke_mask = create_smoke_mask(hsv)

    print("Fire segmentation      : COMPLETE")
    print("Smoke segmentation     : COMPLETE")


    # ==================================================
    # 6. MORPHOLOGICAL PROCESSING
    # ==================================================

    cleaned_fire_mask = clean_mask(fire_mask)

    cleaned_smoke_mask = clean_mask(smoke_mask)

    print("Morphological processing: COMPLETE")


    # ==================================================
    # 7. REGION DETECTION
    # ==================================================

    fire_regions = find_regions(
        cleaned_fire_mask,
        min_area=500
    )

    smoke_regions = find_regions(
        cleaned_smoke_mask,
        min_area=1000,
        max_area=100000
    )

    print("Region detection        : COMPLETE")


    # ==================================================
    # 8. FIRE CLASSIFICATION
    # ==================================================

    confirmed_fire = 0

    fire_features_list = []

    for i, region in enumerate(fire_regions, start=1):

        features = extract_features(
            region,
            hsv,
            fire_mask
        )

        classification = classify_fire(features)

        fire_features_list.append(
            (region, features, classification)
        )

        if classification == "FIRE":
            confirmed_fire += 1

        # Detailed output only with --verbose
        if args.verbose:

            print("\n" + "-" * 50)
            print(f"FIRE REGION {i}")
            print("-" * 50)

            print(
                f"Area             : "
                f"{features['area']:.2f}"
            )

            print(
                f"Width            : "
                f"{features['width']}"
            )

            print(
                f"Height           : "
                f"{features['height']}"
            )

            print(
                f"Aspect Ratio     : "
                f"{features['aspect_ratio']:.2f}"
            )

            print(
                f"Mean Hue         : "
                f"{features['mean_hue']:.2f}"
            )

            print(
                f"Mean Saturation  : "
                f"{features['mean_saturation']:.2f}"
            )

            print(
                f"Mean Value       : "
                f"{features['mean_value']:.2f}"
            )

            print(
                f"Classification    : "
                f"{classification}"
            )


    # ==================================================
    # 9. SMOKE CLASSIFICATION
    # ==================================================

    smoke_detected = 0

    smoke_results = []

    for i, region in enumerate(smoke_regions, start=1):

        classification = classify_smoke(
            region,
            hsv,
            smoke_mask
        )

        smoke_results.append(
            (region, classification)
        )

        if classification == "SMOKE":
            smoke_detected += 1

        # Detailed output only with --verbose
        if args.verbose:

            aspect_ratio = (
                region["width"] / region["height"]
                if region["height"] > 0
                else 0
            )

            print("\n" + "-" * 50)
            print(f"SMOKE REGION {i}")
            print("-" * 50)

            print(
                f"Area             : "
                f"{region['area']:.2f}"
            )

            print(
                f"Width            : "
                f"{region['width']}"
            )

            print(
                f"Height           : "
                f"{region['height']}"
            )

            print(
                f"Aspect Ratio     : "
                f"{aspect_ratio:.2f}"
            )

            print(
                f"Classification    : "
                f"{classification}"
            )


    # ==================================================
    # 10. FINAL CLASSIFICATION
    # ==================================================

    final_status = determine_final_status(
        confirmed_fire,
        smoke_detected
    )


    # ==================================================
    # 11. CREATE OUTPUT IMAGE
    # ==================================================

    output = resized.copy()


    # --------------------------------------------------
    # Draw FIRE regions
    # --------------------------------------------------

    for region, features, classification in fire_features_list:

        if classification != "FIRE":
            continue

        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3
        )

        label = "FIRE"

        label_y = max(y - 35, 0)

        cv2.rectangle(
            output,
            (x, label_y),
            (x + 90, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (x + 8, max(y - 10, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )


    # --------------------------------------------------
    # Draw SMOKE regions
    # --------------------------------------------------

    for region, classification in smoke_results:

        if classification != "SMOKE":
            continue

        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (200, 200, 200),
            2
        )

        label_y = max(y - 35, 0)

        cv2.rectangle(
            output,
            (x, label_y),
            (x + 105, y),
            (200, 200, 200),
            -1
        )

        cv2.putText(
            output,
            "SMOKE",
            (x + 8, max(y - 10, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )


    # ==================================================
    # 12. STATUS PANEL
    # ==================================================

    panel_height = 100

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], panel_height),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        f"STATUS: {final_status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    cv2.putText(
        output,
        f"Fire: {confirmed_fire}    Smoke: {smoke_detected}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2
    )


    # ==================================================
    # 13. SAVE RESULT
    # ==================================================

    output_directory = os.path.dirname(output_path)

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        output
    )

    if not success:

        print("\nERROR: Could not save output image.")

        return


    # ==================================================
    # 14. FINAL CONSOLE RESULT
    # ==================================================

    print_final_result(
        confirmed_fire,
        smoke_detected,
        final_status,
        output_path
    )


    # ==================================================
    # 15. OPTIONAL DISPLAY
    # ==================================================

    if args.show:

        window_name = "Fire & Smoke Detection"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            window_name,
            output
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()


    print("\nProcessing completed successfully.\n")


if __name__ == "__main__":
    main()