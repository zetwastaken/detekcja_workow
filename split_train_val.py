import os
import shutil
import random

# USTAWIENIA
source_images = "sciezka/do/all_images"  # Gdzie są teraz wszystkie zdjęcia
source_labels = "sciezka/do/all_labels"  # Gdzie są teraz wszystkie txt
dest_folder = "dataset_ready"            # Gdzie ma powstać gotowy dataset
split_ratio = 0.8                        # 80% trening, 20% walidacja

# Tworzenie struktur folderów
for split in ['train', 'valid']:
    os.makedirs(f"{dest_folder}/{split}/images", exist_ok=True)
    os.makedirs(f"{dest_folder}/{split}/labels", exist_ok=True)

# Pobranie listy plików (zakładamy, że nazwy plików zdjęć i txt są takie same poza rozszerzeniem)
files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(files) # Mieszamy losowo!

split_index = int(len(files) * split_ratio)
train_files = files[:split_index]
valid_files = files[split_index:]

def copy_files(file_list, split_name):
    for filename in file_list:
        name_no_ext = os.path.splitext(filename)[0]
        
        # Kopiowanie zdjęcia
        shutil.copy(os.path.join(source_images, filename), 
                    os.path.join(dest_folder, split_name, 'images', filename))
        
        # Kopiowanie labela (txt)
        label_file = name_no_ext + ".txt"
        # Sprawdzamy czy label istnieje (czasem są puste zdjęcia bez labeli)
        if os.path.exists(os.path.join(source_labels, label_file)):
            shutil.copy(os.path.join(source_labels, label_file), 
                        os.path.join(dest_folder, split_name, 'labels', label_file))

print("Kopiowanie plików...")
copy_files(train_files, 'train')
copy_files(valid_files, 'valid')
print("Gotowe! Struktura utworzona w folderze 'dataset_ready'")