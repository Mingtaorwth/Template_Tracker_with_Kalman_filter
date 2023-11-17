import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, video_path):
        # Variables
        self.selected_region = None
        self.roi = None
        self.frame_count = 0
        self.update_interval = 1  # Step of refresh the template

        # Kalman Filter
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 Status, 2 Observations
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Get video
        self.video_capture = cv2.VideoCapture(video_path)

        # Select tracking template by mouse
        cv2.namedWindow('Select Region')
        cv2.setMouseCallback('Select Region', self.select_region)

        # Read first frame
        ret, self.frame = self.video_capture.read()

        # Display fierst frame, then press Enter to next step
        cv2.imshow('Select Region', self.frame)
        cv2.waitKey(0)

        # Check the region
        if self.selected_region is None or len(self.selected_region) != 2:
            print("The template has not been selected, stop running")
            cv2.destroyAllWindows()
            exit()

        # Template
        self.roi = self.frame[self.selected_region[0][1]:self.selected_region[1][1],
                   self.selected_region[0][0]:self.selected_region[1][0]]

        # Initializing Kalman Filter
        self.kalman.statePre = np.array([self.selected_region[0][0], self.selected_region[0][1], 0, 0], np.float32)
        self.kalman.statePost = np.array([self.selected_region[0][0], self.selected_region[0][1], 0, 0], np.float32)

        # choose Template Matching method
        self.method = cv2.TM_CCOEFF

    def select_region(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_region = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            # Draw Reclangle
            self.selected_region.append((x, y))
            cv2.rectangle(self.frame, self.selected_region[0], self.selected_region[1], (0, 255, 0), 2)
            cv2.imshow('Select Region', self.frame)

    def run_tracking(self):
        while True:
            ret, self.frame = self.video_capture.read()

            if not ret:
                break

            # convert current frame to grayscale
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)

            # macth template
            result = cv2.matchTemplate(frame_gray, self.roi, self.method)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            # Get the coordinate of mathing result
            top_left = max_loc
            bottom_right = (top_left[0] + self.roi.shape[1], top_left[1] + self.roi.shape[0])

            # Refresh template by update interval
            if self.frame_count % self.update_interval == 0:
                self.roi = self.frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Use Kalman Filter to estimate the coordinate
            self.kalman.correct(np.array([[top_left[0]], [top_left[1]]], np.float32))

            prediction = self.kalman.predict()
            predicted_state = (int(prediction[0]), int(prediction[1]))

            # Draw rectangle in current frame
            cv2.rectangle(self.frame, top_left, bottom_right, (0, 255, 0), 2)  # Measurement
            cv2.rectangle(self.frame, predicted_state,
                          (predicted_state[0] + self.roi.shape[1], predicted_state[1] + self.roi.shape[0]),
                          (0, 0, 255), 2)  # Prediction
            cv2.rectangle(self.frame, (int(self.kalman.statePre[0]), int(self.kalman.statePre[1])),
                          (int(self.kalman.statePre[0] + self.roi.shape[1]),
                           int(self.kalman.statePre[1] + self.roi.shape[0])), (255, 0, 0), 2)  # Correction

            # Display Result
            cv2.imshow('Tracking', self.frame)

            # Refresh frame number
            self.frame_count += 1

            # press ESC to stop running
            if cv2.waitKey(30) == 27:
                break

        # Release
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ObjectTracker("E:/RWTH/Project/YoloV8-detecor-main/Test_videos/Montage R51_Fa Medenus.mp4")
    tracker.run_tracking()
