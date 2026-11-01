"""Selected annotation fields and the ten formal task names."""

FEATURE_FIELDS = [
    "Major Location", "Minor Location", "Second Minor Location", "Contact",
    "Handshape", "Selected Fingers", "Flexion", "Flexion Change", "Spread",
    "Spread Change", "Thumb Position", "Thumb Contact", "Sign Type",
    "Path Movement", "Repeated Movement", "Nondominant Handshape", "Wrist Twist",
]

TASKS = [
    "same_sign", "one_feature_pair", "distinguishing_feature",
    "exists_one_feature_pair", "universal_shared_feature", "feature_value_counting",
    "constraint_satisfaction", "feature_transformation", "common_neighbor",
    "isolate_detection",
]
