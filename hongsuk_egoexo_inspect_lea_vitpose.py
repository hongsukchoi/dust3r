from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS

body_order = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear", 
    "left_big_toe", "left_small_toe", "left_heel", 
    "right_big_toe", "right_small_toe", "right_heel"
] # neck and mid_hip are not in the COCO_WHOLEBODY_KEYPOINTS

# Define the order of left hand keypoints
left_hand_order = [
    "left_wrist_openpose", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb",
    "left_index1", "left_index2", "left_index3", "left_index",
    "left_middle1", "left_middle2", "left_middle3", "left_middle",
    "left_ring1", "left_ring2", "left_ring3", "left_ring",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky"
] # the left_wrist_openpose is not in the COCO_WHOLEBODY_KEYPOINTS

# Define the order of right hand keypoints
right_hand_order = [
    "right_wrist_openpose", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb",
    "right_index1", "right_index2", "right_index3", "right_index",
    "right_middle1", "right_middle2", "right_middle3", "right_middle",
    "right_ring1", "right_ring2", "right_ring3", "right_ring",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky"
] # the right_wrist_openpose is not in the COCO_WHOLEBODY_KEYPOINTS

face = COCO_WHOLEBODY_KEYPOINTS[23:-42]
face_keypoints_order = face[17: 17 + 51]
face_contour_keypoints_order = face[:17]




lea_joint_names = body_order + left_hand_order + right_hand_order + face_keypoints_order + face_contour_keypoints_order
# Find joint names that exist in both lists
common_joint_names = [joint for joint in lea_joint_names if joint in COCO_WHOLEBODY_KEYPOINTS]
print(f"\nNumber of common joints between lea_joint_names and COCO_WHOLEBODY_KEYPOINTS: {len(common_joint_names)}")
print("\nCommon joints:")
print(common_joint_names)
# # Find joint names that are in lea_joint_names but not in COCO_WHOLEBODY_KEYPOINTS
# missing_joints = [joint for joint in lea_joint_names if joint not in COCO_WHOLEBODY_KEYPOINTS]
# print("\nJoints in lea_joint_names but not in COCO_WHOLEBODY_KEYPOINTS:")
# print(missing_joints)

# selected_joint_names = [joint_name for idx, joint_name in enumerate(lea_joint_names) if idx in list(range(46,60))]
# print(selected_joint_names)


print(f"Length of joint_order: {len(lea_joint_names)}")
print(lea_joint_names)