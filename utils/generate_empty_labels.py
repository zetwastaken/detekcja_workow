import os

images_dir = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/tiling/choosen_V1'
labels_dir = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/YOLOv8_segmentation/labels/train'

images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

count = 0
for img_name in images:
    txt_name = os.path.splitext(img_name)[0] + '.txt'
    txt_path = os.path.join(labels_dir, txt_name)

    if not os.path.exists(txt_path):
        with open(txt_path, 'w') as f:
            pass  # create an empty file
        count += 1

print(f"Created {count} empty label files.")