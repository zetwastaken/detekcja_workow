import os
import shutil
import random
from pathlib import Path

# ================= KONFIGURACJA =================
# Ścieżki do Twoich SUROWYCH danych (tam gdzie jest wszystko razem)
source_images = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/tiling/choosen_V1'
source_labels = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/YOLOv8_segmentation/labels/train'


# Gdzie ma powstać GOTOWY dataset (folder zostanie utworzony)
dest_folder = "./datasets/dataset_yolov8_V1"

# Proporcje (0.8 = 80% trening, 20% walidacja)
train_ratio = 0.8 
# ================================================

def split_and_copy():
    # 1. Tworzenie struktury folderów docelowych
    for split in ['train', 'valid']:
        os.makedirs(os.path.join(dest_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, split, 'labels'), exist_ok=True)

    # 2. Pobieranie listy zdjęć
    # Obsługuje jpg, png, jpeg.
    images = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sprawdzenie czy są zdjęcia
    if not images:
        print("BŁĄD: Nie znaleziono zdjęć w folderze źródłowym!")
        return

    # 3. Mieszanie (Shuffle) - KLUCZOWE!
    random.seed(42) # Ustawiamy ziarno, żeby wynik był powtarzalny przy ponownym uruchomieniu
    random.shuffle(images)

    # 4. Obliczanie punktu podziału
    split_index = int(len(images) * train_ratio)
    train_imgs = images[:split_index]
    valid_imgs = images[split_index:]

    print(f"Całkowita liczba zdjęć: {len(images)}")
    print(f"Trening: {len(train_imgs)} | Walidacja: {len(valid_imgs)}")

    # Funkcja pomocnicza do kopiowania
    def process_files(file_list, split_name):
        count_ok = 0
        count_empty_created = 0
        
        print(f"\nPrzetwarzanie zbioru: {split_name}...")
        
        for filename in file_list:
            # Ścieżki źródłowe
            src_img_path = os.path.join(source_images, filename)
            
            # Ustalanie nazwy labela (zamiana rozszerzenia na .txt)
            name_no_ext = os.path.splitext(filename)[0]
            label_name = name_no_ext + ".txt"
            src_label_path = os.path.join(source_labels, label_name)
            
            # Ścieżki docelowe
            dst_img_path = os.path.join(dest_folder, split_name, 'images', filename)
            dst_label_path = os.path.join(dest_folder, split_name, 'labels', label_name)

            # A. Kopiowanie zdjęcia
            shutil.copy2(src_img_path, dst_img_path)

            # B. Obsługa Labela
            if os.path.exists(src_label_path):
                # Jeśli plik txt istnieje - kopiujemy go
                shutil.copy2(src_label_path, dst_label_path)
                count_ok += 1
            else:
                # Jeśli plik txt NIE istnieje - tworzymy pusty (Negative Sample)
                with open(dst_label_path, 'w') as f:
                    pass
                count_empty_created += 1

        print(f"--> Zakończono {split_name}.")
        print(f"    Skopiowano etykiet: {count_ok}")
        print(f"    Utworzono pustych: {count_empty_created} (Negative Samples)")

    # 5. Uruchomienie procesu
    process_files(train_imgs, 'train')
    process_files(valid_imgs, 'valid')
    
    print("\n✅ GOTOWE! Twój dataset jest w folderze:", dest_folder)

if __name__ == "__main__":
    split_and_copy()