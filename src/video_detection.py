import cv2
import numpy as np

from src.preprocessing import resize_image, apply_blur
from src.segmentation import (
    create_fire_mask,
    create_smoke_mask
)
from src.morphology import clean_mask
from src.detection import find_regions
from src.classification import (
    extract_features,
    classify_fire,
    classify_smoke,
    determine_final_status
)


def process_frame(
    frame,
    previous_gray=None
):
    """
    Process one video frame.

    Returns:
        output_frame
        status
        fire_count
        smoke_count
        current_gray
    """

    frame = resize_image(frame, width=800)

    blurred = apply_blur(frame)

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    fire_mask = create_fire_mask(hsv)
    smoke_mask = create_smoke_mask(hsv)

    cleaned_fire_mask = clean_mask(
        fire_mask
    )

    cleaned_smoke_mask = clean_mask(
        smoke_mask
    )

    fire_regions = find_regions(
        cleaned_fire_mask,
        min_area=500
    )

    smoke_regions = find_regions(
        cleaned_smoke_mask,
        min_area=1000,
        max_area=100000
    )

    fire_count = 0
    smoke_count = 0

    output_frame = frame.copy()

    # --------------------------------------------------
    # FIRE DETECTION
    # --------------------------------------------------

    for region in fire_regions:

        features = extract_features(
            region,
            hsv,
            cleaned_fire_mask
        )

        result = classify_fire(
            features
        )

        if result == "FIRE":

            fire_count += 1

            x = region["x"]
            y = region["y"]
            width = region["width"]
            height = region["height"]

            cv2.rectangle(
                output_frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                "FIRE",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------
    # SMOKE DETECTION
    # --------------------------------------------------

    for region in smoke_regions:

        smoke_result = classify_smoke(
            region,
            hsv,
            cleaned_smoke_mask
        )

        if smoke_result != "SMOKE":
            continue

        # --------------------------------------------------
        # TEMPORAL MOTION CHECK
        # --------------------------------------------------

        motion_detected = False

        if previous_gray is not None:

            difference = cv2.absdiff(
                gray,
                previous_gray
            )

            x = region["x"]
            y = region["y"]
            width = region["width"]
            height = region["height"]

            roi_difference = difference[
                y:y + height,
                x:x + width
            ]

            if roi_difference.size > 0:

                mean_difference = float(
                    np.mean(roi_difference)
                )

                # Smoke should show some temporal change.
                if mean_difference >= 2.0:
                    motion_detected = True

        if motion_detected:

            smoke_count += 1

            x = region["x"]
            y = region["y"]
            width = region["width"]
            height = region["height"]

            cv2.rectangle(
                output_frame,
                (x, y),
                (x + width, y + height),
                (180, 180, 180),
                2
            )

            cv2.putText(
                output_frame,
                "SMOKE",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2
            )

    status = determine_final_status(
        fire_count,
        smoke_count
    )

    # --------------------------------------------------
    # STATUS PANEL
    # --------------------------------------------------

    cv2.rectangle(
        output_frame,
        (0, 0),
        (output_frame.shape[1], 90),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output_frame,
        f"STATUS: {status}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Fire: {fire_count}    Smoke: {smoke_count}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2
    )

    return (
        output_frame,
        status,
        fire_count,
        smoke_count,
        gray
    )


def process_video(
    input_path,
    output_path,
    show=False
):
    """
    Process a complete video frame by frame.

    Fire and smoke detections must persist for several
    consecutive frames before being counted as confirmed.
    Smoke candidates must also show temporal movement.
    """

    video = cv2.VideoCapture(
        input_path
    )

    if not video.isOpened():

        print(
            f"ERROR: Could not open video: "
            f"{input_path}"
        )

        return False

    print("\n" + "=" * 60)
    print(
        "          FIRE & SMOKE DETECTION SYSTEM"
    )
    print(
        "              COMPUTER VISION PROJECT"
    )
    print("=" * 60)

    print(
        f"\nInput video : {input_path}"
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    frame_width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if frame_width <= 0 or frame_height <= 0:

        video.release()

        print(
            "\nERROR: Invalid video dimensions."
        )

        return False

    print(
        "Video loading           : COMPLETE"
    )

    output_width = 800

    output_height = int(
        frame_height *
        (output_width / frame_width)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            output_width,
            output_height
        )
    )

    if not writer.isOpened():

        video.release()

        print(
            "\nERROR: Could not create output video."
        )

        return False

    print(
        "Video preprocessing     : READY"
    )

    print(
        "Fire segmentation       : READY"
    )

    print(
        "Smoke segmentation      : READY"
    )

    print(
        "Morphological processing: READY"
    )

    print(
        "Region detection        : READY"
    )

    print(
        "Temporal confirmation   : 5 consecutive frames"
    )

    print(
        "Smoke motion validation : ENABLED"
    )

    frame_count = 0

    fire_frames = 0
    smoke_frames = 0

    fire_streak = 0
    smoke_streak = 0

    confirmation_frames = 5

    previous_gray = None

    while True:

        success, frame = video.read()

        if not success:
            break

        (
            output_frame,
            frame_status,
            fire_count,
            smoke_count,
            current_gray
        ) = process_frame(
            frame,
            previous_gray
        )

        previous_gray = current_gray

        # --------------------------------------------------
        # TEMPORAL CONFIRMATION
        # --------------------------------------------------

        if fire_count > 0:
            fire_streak += 1
        else:
            fire_streak = 0

        if smoke_count > 0:
            smoke_streak += 1
        else:
            smoke_streak = 0

        confirmed_fire = (
            fire_streak >= confirmation_frames
        )

        confirmed_smoke = (
            smoke_streak >= confirmation_frames
        )

        if confirmed_fire:
            fire_frames += 1

        if confirmed_smoke:
            smoke_frames += 1

        if (
            confirmed_fire
            and confirmed_smoke
        ):
            temporal_status = "FIRE + SMOKE"

        elif confirmed_fire:
            temporal_status = "FIRE"

        elif confirmed_smoke:
            temporal_status = "SMOKE"

        else:
            temporal_status = "NORMAL"

        # --------------------------------------------------
        # FINAL STATUS PANEL
        # --------------------------------------------------

        cv2.rectangle(
            output_frame,
            (0, 0),
            (
                output_frame.shape[1],
                90
            ),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            output_frame,
            f"STATUS: {temporal_status}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Fire: {fire_count}    Smoke: {smoke_count}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2
        )

        writer.write(
            output_frame
        )

        frame_count += 1

        if show:

            cv2.imshow(
                "Fire & Smoke Detection - Video",
                output_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    video.release()
    writer.release()

    if show:
        cv2.destroyAllWindows()

    # --------------------------------------------------
    # FINAL VIDEO STATUS
    # --------------------------------------------------

    if (
        fire_frames > 0
        and smoke_frames > 0
    ):
        video_status = "FIRE + SMOKE"

    elif fire_frames > 0:
        video_status = "FIRE"

    elif smoke_frames > 0:
        video_status = "SMOKE"

    else:
        video_status = "NORMAL"

    print("\n" + "-" * 60)
    print(
        "                    FINAL RESULT"
    )
    print("-" * 60)

    print(
        f"Frames processed       : {frame_count}"
    )

    print(
        f"Fire frames detected   : {fire_frames}"
    )

    print(
        f"Smoke frames detected  : {smoke_frames}"
    )

    print(
        f"Detection status       : {video_status}"
    )

    print("-" * 60)

    print(
        f"Output saved           : {output_path}"
    )

    print("-" * 60)

    return True