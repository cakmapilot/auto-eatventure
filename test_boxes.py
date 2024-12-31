from adb_autoplay import AutoEatventure, Timer


if __name__ == "__main__":
    dev = AutoEatventure()
    dev.start_app()

    # to check if game is playable.
    dev.init_game()

    with Timer("Finding boxes"):
        # boxes find and click
        print('finding boxes')
        dev.capture_screenshot()
        # imshow(dev.matching_templates_cv2["box"]["simple"], False)
        # imshow(dev.current_cv2_sc)
        # imshow(dev.current_cv2_sc_grayscale)
        # imshow(dev.current_cv2_sc_bgr2hsv)
        dev.open_boxes()