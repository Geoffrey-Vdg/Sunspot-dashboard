import os
import json
import shutil

def main():
    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # Update the source directory to the actual location of your labeled images
    images_source_dir = os.path.join(project_dir, "dataset", "images", "default")
    supervised_dir = os.path.join(project_dir, "supervised")
    annotations_dir = os.path.join(supervised_dir, "annotations")
    images_dir = os.path.join(supervised_dir, "images")
    labeled_images_output_dir = os.path.join(images_dir, "default")
    annotations_file_source = os.path.join(project_dir, "dataset", "annotations", "instances_default.json")
    annotations_file_destination = os.path.join(annotations_dir, "instances_default.json")

    # Ensure the directory structure exists
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(labeled_images_output_dir, exist_ok=True)

    # Clear previous images and annotations
    if os.path.exists(labeled_images_output_dir):
        print(f"Clearing existing images in {labeled_images_output_dir}...")
        shutil.rmtree(labeled_images_output_dir)
        os.makedirs(labeled_images_output_dir, exist_ok=True)
    if os.path.exists(annotations_file_destination):
        print(f"Removing existing annotations file at {annotations_file_destination}...")
        os.remove(annotations_file_destination)

    # Load annotations file
    if not os.path.exists(annotations_file_source):
        print(f"Annotations file not found at {annotations_file_source}.")
        return

    with open(annotations_file_source, 'r') as f:
        try:
            annotations = json.load(f)
        except json.JSONDecodeError:
            print(f"Invalid JSON format in {annotations_file_source}. Exiting.")
            return

    # Validate annotations data
    if not annotations.get("images") or not annotations.get("annotations"):
        print("Annotations file is empty or missing required data.")
        return

    # Map image IDs to filenames
    image_id_to_filename = {
        img['id']: img['file_name'] for img in annotations['images']
    }

    # Get the IDs of labeled images
    labeled_image_ids = {anno['image_id'] for anno in annotations['annotations']}

    # Debug output: List image IDs and annotations
    print(f"Total images in annotations: {len(annotations['images'])}")
    print(f"Total annotations: {len(annotations['annotations'])}")
    print(f"Total labeled image IDs: {len(labeled_image_ids)}")

    # Copy only labeled images to the new directory
    copied_images_count = 0
    for image_id in labeled_image_ids:
        filename = image_id_to_filename.get(image_id)
        if filename:
            src = os.path.join(images_source_dir, filename)
            dst = os.path.join(labeled_images_output_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
                copied_images_count += 1
            else:
                print(f"Image file {src} not found. Skipping.")
        else:
            print(f"No filename found for image ID {image_id}. Skipping.")

    print(f"{copied_images_count} labeled images have been copied to: {labeled_images_output_dir}")

    # Update annotations to include only the copied images
    filtered_images = [img for img in annotations['images'] if img['id'] in labeled_image_ids]
    filtered_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] in labeled_image_ids]

    # Create the new annotations dictionary
    new_annotations = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": annotations.get("categories", [])
    }

    # Save the new annotations to the destination file
    with open(annotations_file_destination, 'w') as f:
        json.dump(new_annotations, f, indent=4)

    print(f"Annotations file has been updated and saved to {annotations_file_destination}")

if __name__ == "__main__":
    main()
