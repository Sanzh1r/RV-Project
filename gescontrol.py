import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import mediapipe as mp
import time

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        self.get_logger().info('Starting GestureControlNode (HAND GESTURE CONTROL)')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.process_frame)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera!')
            rclpy.shutdown()

        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Configuration parameters
        self.linear_speed = 0.27
        self.angular_speed = 1.0
        self.stop_distance = 300 

        # Robot state
        self.last_cmd = Twist()
        
        # ZigZag parameters
        self.zigzag_start_time = None
        self.zigzag_phase_duration = 2.0
        self.zigzag_forward_speed = 0.3
        self.zigzag_turn_speed = 1.2
        self.zigzag_active = False

    def detect_gesture(self, hand_landmarks, frame_shape):
        """Detect gesture based on hand landmarks"""
        height, width, _ = frame_shape
        
        # Get fingertip positions
        fingertips = []
        finger_indices = [4, 8, 12, 16, 20]
        for idx in finger_indices:
            x = int(hand_landmarks.landmark[idx].x * width)
            y = int(hand_landmarks.landmark[idx].y * height)
            fingertips.append((x, y))
        
        # Get palm position (using wrist landmark)
        palm_x = int(hand_landmarks.landmark[0].x * width)
        palm_y = int(hand_landmarks.landmark[0].y * height)
        
        # Get middle of palm (average of specific landmarks)
        palm_center_x = int(sum([hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]]) / 5 * width)
        palm_center_y = int(sum([hand_landmarks.landmark[i].y for i in [0, 5, 9, 13, 17]]) / 5 * height)
        
        # Calculate palm size (distance from wrist to middle finger base)
        middle_base = (int(hand_landmarks.landmark[9].x * width), 
                       int(hand_landmarks.landmark[9].y * height))
        palm_size = np.sqrt((palm_x - middle_base[0])**2 + (palm_y - middle_base[1])**2)
        
        # Horizontal position of palm relative to frame center
        center_offset = palm_center_x - width // 2
        normalized_offset = center_offset / (width // 2)  # -1 to 1
        
        # Vertical position of palm (used for forward/backward)
        vertical_position = palm_center_y / height  # 0 to 1
        
        fingers_up = []
        
        # Special check for thumb - based on horizontal position relative to thumb base
        thumb_tip_x = hand_landmarks.landmark[4].x
        thumb_base_x = hand_landmarks.landmark[2].x
        if (thumb_tip_x < thumb_base_x):
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # For other fingers - check if fingertip is above finger pip
        for idx in range(1, 5):
            fingertip_y = hand_landmarks.landmark[finger_indices[idx]].y
            pip_y = hand_landmarks.landmark[finger_indices[idx] - 2].y
            if fingertip_y < pip_y: 
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Detect gestures based on finger states
        gesture = "unknown"
        
        if sum(fingers_up) == 0:
            gesture = "stop"
        
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = "forward"
        
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = "backward"
        
        elif fingers_up == [1, 0, 0, 0, 0]:
            gesture = "turn_left"
            
        elif fingers_up == [0, 0, 0, 0, 1]:
            gesture = "turn_right"
            
        elif fingers_up == [0, 1, 1, 1, 0]:
            gesture = "position_control"
        
        elif fingers_up == [0, 1, 0, 0, 1]:
            gesture = "zigzag"

            
        return {
            "gesture": gesture,
            "palm_position": (palm_center_x, palm_center_y),
            "palm_size": palm_size,
            "normalized_offset": normalized_offset,
            "vertical_position": vertical_position,
            "fingers_up": fingers_up
        }

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame from camera')
            return
            
        # Flip image horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        twist = Twist()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first detected hand
            gesture_info = self.detect_gesture(hand_landmarks, frame.shape)
            
            gesture = gesture_info["gesture"]
            palm_size = gesture_info["palm_size"]
            normalized_offset = gesture_info["normalized_offset"]
            vertical_position = gesture_info["vertical_position"]
            
            self.get_logger().info(f'Detected gesture: {gesture}, palm size: {palm_size:.1f}')
            
            # Handle different gestures
            if gesture == "stop":
                # Stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
                
            elif gesture == "forward":
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
                
            elif gesture == "backward":
                twist.linear.x = -self.linear_speed
                twist.angular.z = 0.0
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
                
            elif gesture == "turn_left":
                twist.linear.x = 0.0
                twist.angular.z = -self.angular_speed
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
                
            elif gesture == "turn_right":
                twist.linear.x = 0.0
                twist.angular.z = self.angular_speed
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
                
            elif gesture == "position_control":
                # Control direction based on hand position
                twist.linear.x = self.linear_speed * (1.0 - vertical_position * 2.0)
                twist.angular.z = -self.angular_speed * normalized_offset 
                # Reset zigzag state
                self.zigzag_active = False
                self.zigzag_start_time = None
            
            elif gesture == "zigzag":
                # Start zigzag pattern if not already active
                if not self.zigzag_active:
                    self.zigzag_start_time = time.time()
                    self.zigzag_active = True
                
                # Calculate how far we are in the zigzag cycle
                current_time = time.time()
                elapsed_time = current_time - self.zigzag_start_time
                
                # Use palm size to adjust zigzag parameters
                # Larger palm (closer to camera) = tighter zigzag
                turn_intensity = min(1.5, max(0.5, palm_size / 300))
                
                # Use vertical position to adjust forward speed
                speed_adjustment = 1.0 - (vertical_position * 0.5)
                
                # Calculate which phase of zigzag we're in
                phase = (elapsed_time % (self.zigzag_phase_duration * 2)) / self.zigzag_phase_duration
                
                if phase < 1.0:
                    # Turn left and move forward
                    twist.linear.x = self.zigzag_forward_speed * speed_adjustment
                    twist.angular.z = self.zigzag_turn_speed * turn_intensity
                    self.get_logger().info('ZigZag: Left turn phase')
                else:
                    # Turn right and move forward
                    twist.linear.x = self.zigzag_forward_speed * speed_adjustment
                    twist.angular.z = -self.zigzag_turn_speed * turn_intensity
                    self.get_logger().info('ZigZag: Right turn phase')
                
            else:
                # Unknown gesture - keep last command for stability
                twist = self.last_cmd
        else:
            # No hand detected - stop the robot
            self.get_logger().info('No hand detected')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # Reset zigzag state
            self.zigzag_active = False
            self.zigzag_start_time = None
        
        self.publisher.publish(twist)
        self.last_cmd = twist
        
        # Debug visualization
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        # Display the resulting frame
        cv2.imshow('Hand Gesture Control', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
