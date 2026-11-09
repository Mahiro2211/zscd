"""Text prompts for zero-shot semantic-consistency change detection.

Class system follows Zan et al. 2025 (WHU-GCD): 6 land-cover categories
(building / road / water / barren / forest / agriculture), see notes_paper.md.
"""

# WHU-GCD 6 land-cover classes (paper Table 1); farmland == "agriculture"
LAND_COVER_CLASSES = ['building', 'road', 'water', 'bare land', 'forest', 'farmland']

# prompt templates: {} is replaced by the class name
TEMPLATES = [
    'a satellite image of a {}',
    'an aerial image of a {}',
    'a remote sensing image of a {}',
    'a photo of a {} from a bird\'s-eye view',
]
