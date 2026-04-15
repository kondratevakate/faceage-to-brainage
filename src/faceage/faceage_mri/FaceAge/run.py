import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import gc
import numpy as np
import tensorflow as tf
from skimage.io import imsave, imread
import mtcnn
import keras
import PIL



def get_face_bbox_from_image(path_to_image, detector_model):
    # sanity check
    assert os.path.exists(path_to_image)

    pat_img = imread(path_to_image)

    # Strip alpha channel if it's RGBA
    if len(pat_img.shape) == 3 and pat_img.shape[2] == 4:
        pat_img = pat_img[:, :, :3]

    try:
        results = detector_model.detect_faces(pat_img)
        if len(results) > 0:
            return results[0]
        else:
            return dict()
    except Exception as e:
        print(f'ERROR: Processing error for file "{path_to_image}": {e}')
        return dict()

def get_model_prediction(model, path_to_image, mtcnn_output_dict):

  # sanity check
  assert os.path.exists(path_to_image)

  pat_img = imread(path_to_image)

  # extract the bounding box from the first face
  x1, y1, width, height = mtcnn_output_dict['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height

  # crop the face
  pat_face = pat_img[y1:y2, x1:x2]

  # resize cropped image to the model input size
  pat_face_pil = PIL.Image.fromarray(np.uint8(pat_face)).convert('RGB')
  pat_face = np.asarray(pat_face_pil.resize((160, 160)))

  # prep image for TF processing
  mean, std = pat_face.mean(), pat_face.std()
  pat_face = (pat_face - mean) / std
  pat_face_input = pat_face.reshape(1, 160, 160, 3)

  return np.squeeze(model.predict(pat_face_input))


if __name__ == "__main__":

    # --- Configuration ---
    input_base_path = "/workspace/faceage_mri/renders_filtered"
    output_crop_path = "/workspace/faceage_mri/renders_cropped"
    N_SUBJECTS = 5

    BASE_MODEL_PATH = "/workspace/faceage_mri/FaceAge/weights"

    # Create the output directory if it doesn't exist
    os.makedirs(output_crop_path, exist_ok=True)

    input_file_list = [f for f in os.listdir(input_base_path) if f.endswith(('.png', '.jpg'))]
    if N_SUBJECTS > 0:
        input_file_list = input_file_list[:N_SUBJECTS]

    face_bbox_dict = dict()
    t = time.time()

    print(f"Found {len(input_file_list)} images to process.")

    # Initialize MTCNN ONCE outside the loop to prevent memory leaks
    print("Initializing MTCNN model...")
    detector = mtcnn.mtcnn.MTCNN()

    for idx, input_image in enumerate(input_file_list):
        
        # get rid of label information and file extension
        subj_id = input_image.split(".")[0]
        print(f'\n({idx + 1}/{len(input_file_list)}) Running face localization for "{subj_id}"')

        path_to_image = os.path.join(input_base_path, input_image)
        face_bbox_dict[subj_id] = dict()
        face_bbox_dict[subj_id]["path_to_image"] = path_to_image

        # Get the bounding box
        bbox_dict = get_face_bbox_from_image(path_to_image, detector)
        face_bbox_dict[subj_id]["mtcnn_output_dict"] = bbox_dict

        # --- NEW: Crop and Save Logic ---
        if bbox_dict:
            # Extract coordinates
            x1, y1, width, height = bbox_dict['box']
            
            # MTCNN can sometimes return negative coordinates if the face is cut off at the edge
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = x1 + width, y1 + height

            # Load the original image specifically for cropping
            img_orig = imread(path_to_image)
            
            # Ensure we are saving standard RGB (strip Alpha)
            if len(img_orig.shape) == 3 and img_orig.shape[2] == 4:
                img_orig = img_orig[:, :, :3]

            # Crop the face array
            img_face = img_orig[y1:y2, x1:x2]

            # Define the save path and save it
            save_path = os.path.join(output_crop_path, f"{subj_id}_crop.png")
            
            # Check if the cropped image is valid before saving (prevents 0-pixel height/width crashes)
            if img_face.size > 0:
                imsave(save_path, img_face)
                print(f'  -> Success! Face cropped and saved to: {save_path}')
            else:
                print('  -> ERROR: Invalid crop dimensions.')
        else:
            print('  -> WARNING: No face detected. Skipping crop.')

        # solves known TF memory leaks for the MTCNN pipeline
        if not (idx + 1) % 5:
            tf.keras.backend.clear_session()
            gc.collect()

    print(f"\nPipeline finished in {time.time() - t:.2f} seconds.")
    print(f"Check the '{output_crop_path}' directory for your cropped images.")

    model_path = os.path.join(BASE_MODEL_PATH, "faceage_model.h5")
    model = keras.models.load_model(model_path)


    age_pred_dict = dict()

    t = time.time()

    print()

    for idx, subj_id in enumerate(face_bbox_dict.keys()):

        print('(%g/%g) Running the age estimation step for "%s"'%(idx + 1,
                                                                    len(face_bbox_dict),
                                                                    subj_id))

        path_to_image = face_bbox_dict[subj_id]["path_to_image"]
        mtcnn_output_dict = face_bbox_dict[subj_id]["mtcnn_output_dict"]

        age_pred_dict[subj_id] = dict()

        age_pred_dict[subj_id]["faceage"] = get_model_prediction(model, path_to_image, mtcnn_output_dict)
        # age_pred_dict[subj_id]["age"] = face_bbox_dict[subj_id]["age"]
        # age_pred_dict[subj_id]["gender"] = face_bbox_dict[subj_id]["gender"]
        # age_pred_dict[subj_id]["race"] = face_bbox_dict[subj_id]["race"]


    print(f"age_pred_dict: {age_pred_dict}")


