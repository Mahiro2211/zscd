"""zscd: Zero-Shot domain-generalization Change Detection with frozen CLIP.

Pipeline: zero-training pixel-text similarity / feature-difference scores
(semantic / notext / absdiff tracks), fixed global threshold 0.5, evaluated on
the out-of-domain WHU-GCD test set II (~/Dataset/test2).
"""
