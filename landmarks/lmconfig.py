LANDMARK_TARGET = 'multi_channel'
# LANDMARK_TARGET = 'hamming'
MIN_LANDMARK_CONF = 0.8

LANDMARK_OCULAR_NORM = 'outer'
PREDICT_HEATMAP = True
HEATMAP_SIZE = 128

LANDMARKS_6 = [36, 39, 42, 45, 48, 54]
LANDMARKS_9 = [30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_12 = [21, 22, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_19 = [0, 4, 8, 12, 16, 17, 21, 22, 26, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_22 = [0, 4, 8, 12, 16, 17, 21, 22, 26, 27, 28, 29, 30, 36, 39, 42, 45, 48, 51, 54, 57]
LANDMARKS_14 = [17, 26, 21, 22, 27, 30, 36, 39, 42, 45, 48, 51, 54, 57]

COARSE_LANDMARKS = range(68)
# COARSE_LANDMARKS = [8, 36, 45, 48, 54]
COARSE_LANDMARKS_TO_ID = {lm: i for i,lm in enumerate(COARSE_LANDMARKS)}


# hack for CVPR
# def config_landmarks(dataset):
#     global LANDMARKS
#     global ALL_LANDMARKS
#     global LANDMARKS_ONLY_OUTLINE
#     global LANDMARKS_NO_OUTLINE
#     global NUM_LANDMARKS
#     global NUM_LANDMARK_HEATMAPS
#     global LANDMARK_ID_TO_HEATMAP_ID
#
#     if dataset == '300w':
#         # 300W
#         ALL_LANDMARKS = list(range(68))
#         LANDMARKS_NO_OUTLINE = list(range(17,68))
#         LANDMARKS_ONLY_OUTLINE = list(range(17))
#     elif dataset == 'aflw':
#         # AFLW
#         ALL_LANDMARKS = list(range(0, 19))
#         LANDMARKS_NO_OUTLINE = list(range(0,19))
#         LANDMARKS_ONLY_OUTLINE = list(range(0,19))
#     elif dataset == 'wflw':
#         # WFLW
#         ALL_LANDMARKS = list(range(0, 98))
#         LANDMARKS_NO_OUTLINE = list(range(33,98))
#         LANDMARKS_ONLY_OUTLINE = list(range(0,33))
#     else:
#         raise ValueError()
#
#     LANDMARKS = ALL_LANDMARKS
#     NUM_LANDMARKS = len(LANDMARKS)

    # NUM_LANDMARK_HEATMAPS = len(LANDMARKS)



