from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import os

if 'tpu' not in os.listdir('../'):
    from git import Repo

    Repo.clone_from('https://github.com/tensorflow/tpu/', '../tpu')

sys.path.insert(0, '../tpu/models/official')
sys.path.insert(0, '../tpu/models/official/mask_rcnn')

session = tf.Session(graph=tf.Graph())
saved_model_dir = 'gs://cloud-tpu-checkpoints/mask-rcnn/1555659850'
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)

sys.path.insert(0, '../tpu/models/official')
sys.path.insert(0, '../tpu/models/official/mask_rcnn')

import coco_metric
from mask_rcnn.object_detection import visualization_utils

ID_MAPPING = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
category_index = {k: {'id': k, 'name': ID_MAPPING[k]} for k in ID_MAPPING}

# Separated classes
# animals: [16:25]
# food: [44:61]
# furniture: [62:82, 85, 86]
# blogs: [1, 27:43]
# cars: [3, 4, 6, 8, 10:14]

keys_animals = []
keys_food = []
keys_furniture = []
keys_blogs_and_people = []
keys_cars = []

for i in ID_MAPPING:
    if i in range(16, 26):
        keys_animals.append(i)
    elif i in range(44, 62):
        keys_food.append(i)
    elif i in list(range(62, 83)) + [85, 86]:
        keys_furniture.append(i)
    elif i in [1] + list(range(27, 44)):
        keys_blogs_and_people.append(i)
    elif i in [3, 4, 6, 8] + list(range(10, 15)):
        keys_cars.append(i)

class_animals = [ID_MAPPING[value] for value in keys_animals]
class_food = [ID_MAPPING[value] for value in keys_food]
class_furniture = [ID_MAPPING[value] for value in keys_furniture]
class_blogs_and_people = [ID_MAPPING[value] for value in keys_blogs_and_people]
class_cars = [ID_MAPPING[value] for value in keys_cars]


def get_images_category(pages_img_array):
    # Create array to segregate and count detected objects into 4 categories
    # 0 element of array - food
    # 1 element of array - people
    # 2 element of array - animals
    # 3 element of array - furniture
    # 4 element of array - cars
    detection_per_img = np.zeros(5, dtype='int32')
    detections_per_img = []

    # Array for full detection items in all pictures
    detection_classes_profile = np.empty(0).astype('int32')

    # As we have different profiles with different values of images
    # we count profile images
    profile_imgs_nums = len(pages_img_array)

    # for-loop for every single img
    for i in range(profile_imgs_nums):
        print(f'Processing image number {i + 1}.')
        image_path = 'test.jpg'

        np_image = pages_img_array[i].astype(np.uint8)
        im = Image.fromarray(np_image)
        im.save(image_path)

        with open(image_path, 'rb') as f:
            np_image_string = np.array([f.read()])

        image = Image.open(image_path)
        width, height = image.size

        # Perform instance segmentation and retrieve the predictions
        num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, image_info = session.run(
            ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0',
             'ImageInfo:0'],
            feed_dict={'Placeholder:0': np_image_string})

        num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
        detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
        detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
        detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
        instance_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
        ymin, xmin, ymax, xmax = np.split(detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
        segmentations = coco_metric.generate_segmentation_from_masks(instance_masks, processed_boxes, height, width)

        # Convert nums_key from ID_MAPPING to words
        detection_classes_words = list(map(lambda num_class: ID_MAPPING[num_class], detection_classes))
        detection_per_img += img_class(detection_classes_words)
        detections_per_img.append(detection_per_img)

        max_boxes_to_draw = 50  # @param {type:"integer"}
        min_score_thresh = 0.1  # @param {type:"slider", min:0, max:1, step:0.01}

        image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
            np_image,
            detection_boxes,
            detection_classes,
            detection_scores,
            category_index,
            instance_masks=segmentations,
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh)
        output_image_path = 'test_results.jpg'
        Image.fromarray(image_with_detections.astype(np.uint8)).save(output_image_path)

    category_message, best_category = classifier_pages(detection_per_img)

    # Save best image
    best_img_i = np.argmax(np.array(detections_per_img)[:, best_category])
    best_img_array = pages_img_array[best_img_i]

    save_best_img(best_img_array)

    return category_message


from collections import Counter


def img_class(img_detection):
    top5 = Counter(img_detection).most_common()[:5]
    detector_count_food = 0
    detector_count_people = 0
    detector_count_animals = 0
    detector_count_furniture = 0
    detector_count_cars = 0
    detector_counter = np.array([detector_count_food, detector_count_people, detector_count_animals,
                                 detector_count_furniture, detector_count_cars], dtype='int32')

    for index in range(len(top5)):
        if [*dict(top5)][index] in class_food:
            detector_counter[0] += 1
        elif [*dict(top5)][index] in class_animals:
            detector_counter[2] += 1
        elif [*dict(top5)][index] in class_blogs_and_people:
            detector_counter[1] += 1
        elif [*dict(top5)][index] in class_furniture:
            detector_counter[3] += 1
        elif [*dict(top5)][index] in class_cars:
            detector_counter[4] += 1
    return detector_counter


# Classifier pages
# 0 element of array - food
# 1 element of array - blog
# 2 element of array - animals
# 3 element of array - furniture
# 4 element of array - cars

def classifier_pages(counter_pages_array):
    all_detected_obj = np.sum(counter_pages_array)
    class_names = {
        0: 'Food',
        1: 'Blog',
        2: 'Animals',
        3: 'Furniture',
        4: 'Cars'
    }
    percent_per_class = np.round(counter_pages_array / all_detected_obj * 100, 2)
    categories = [class_names[i] for i in list(class_names.keys())]
    print(dict(zip(categories, percent_per_class)))
    # class food
    if percent_per_class[0] >= 45:
        print(f'Category of the profile is {class_names[0]}')
        return f'Category of the profile is {class_names[0]}'
    # class food and people
    elif percent_per_class[0] >= 30 and percent_per_class[1] >= 20:
        print(f'Category of the profile is {class_names[0]} and {class_names[1]}')
        return f'Category of the profile is {class_names[0]} and {class_names[1]}'
    # class animals
    elif np.argmax(percent_per_class) == 2:
        print(f'Category of the profile is {class_names[2]}')
        return f'Category of the profile is {class_names[2]}'
    # class people and animals
    elif percent_per_class[1] >= 25 and percent_per_class[2] >= 20 and np.all(
            percent_per_class[0] + percent_per_class[4] <= 40):
        print(f'Category of the profile is {class_names[1]} and {class_names[2]}')
        return f'Category of the profile is {class_names[1]} and {class_names[2]}'
    # class people
    elif percent_per_class[1] >= 25 and np.all(
            percent_per_class[0] + percent_per_class[2] + percent_per_class[4] <= 40):
        print(f'Category of the profile is {class_names[1]}')
        return f'Category of the profile is {class_names[1]}'

    # class people and furniture
    elif percent_per_class[1] >= 30 and percent_per_class[3] >= 50:
        print(f'Category of the profile is {class_names[1]} and {class_names[3]}')
        return f'Category of the profile is {class_names[1]} and {class_names[3]}'
    # class furniture
    elif percent_per_class[3] >= 50 and np.all(percent_per_class[:3] < 30):
        print(f'Category of the profile is {class_names[3]}')
        return f'Category of the profile is {class_names[3]}'
    # class cars
    elif percent_per_class[4] >= 40:
        print(f'Category of the profile is {class_names[4]}')
        return f'Category of the profile is {class_names[4]}'
    else:
        print('Category was not defined')
        return 'Category was not defined'


def save_best_img(img_np_array):
    image_path = '../flask_app/static/best_original.jpg'

    np_image = img_np_array.astype(np.uint8)
    im = Image.fromarray(np_image)
    im.save(image_path)

    with open(image_path, 'rb') as f:
        np_image_string = np.array([f.read()])

    image = Image.open(image_path)
    width, height = image.size

    # Perform instance segmentation and retrieve the predictions
    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, image_info = session.run(
        ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0',
         'ImageInfo:0'],
        feed_dict={'Placeholder:0': np_image_string})

    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
    instance_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
    ymin, xmin, ymax, xmax = np.split(detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = coco_metric.generate_segmentation_from_masks(instance_masks, processed_boxes, height, width)

    max_boxes_to_draw = 50  # @param {type:"integer"}
    min_score_thresh = 0.1  # @param {type:"slider", min:0, max:1, step:0.01}

    image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
        np_image,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        instance_masks=segmentations,
        use_normalized_coordinates=False,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh)
    output_image_path = '../flask_app/static/best_result.jpg'
    Image.fromarray(image_with_detections.astype(np.uint8)).save(output_image_path)
