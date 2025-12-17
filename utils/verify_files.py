import os

# --- KONFIGURACJA ---
# Wpisz tutaj ścieżki do swoich folderów
images_dir = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/detekcja_workow/tiling/choosen_V1'
labels_dir = 'S:/MyFiles/Studia/Magisterskie/Sem2/Worki/YOLOv8_segmentation/labels/train'
# --------------------

def sprawdz_spojnosc(dir_img, dir_lbl):
    # Pobieramy tylko nazwy plików bez rozszerzeń (np. 'obraz1' zamiast 'obraz1.jpg')
    # filtrujemy, żeby nie brać śmieci systemowych
    set_zdjecia = {os.path.splitext(f)[0] for f in os.listdir(dir_img) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
    set_labele = {os.path.splitext(f)[0] for f in os.listdir(dir_lbl) if f.lower().endswith('.txt')}

    # Obliczamy różnice zbiorów
    # Zdjęcia, które nie mają pliku txt
    brakuje_txt = set_zdjecia - set_labele
    # Pliki txt, które nie mają zdjęcia
    brakuje_img = set_labele - set_zdjecia

    print("-" * 40)
    print(f"RAPORT DLA: {dir_img}")
    print(f"Znaleziono zdjęć:  {len(set_zdjecia)}")
    print(f"Znaleziono labeli: {len(set_labele)}")
    print("-" * 40)

    if not brakuje_txt and not brakuje_img:
        print("✅ SUKCES! Wszystkie pliki pasują do siebie (1:1).")
    else:
        if brakuje_txt:
            print(f"❌ Brakuje plików .txt dla {len(brakuje_txt)} zdjęć (zostaną pominięte w treningu):")
            for name in sorted(brakuje_txt):
                print(f"   - {name}.txt")
            print("")

        if brakuje_img:
            print(f"⚠️  Istnieją pliki .txt bez zdjęć ({len(brakuje_img)} sztuk):")
            for name in sorted(brakuje_img):
                print(f"   - {name}")

# Sprawdzenie czy ścieżki istnieją przed uruchomieniem
if os.path.exists(images_dir) and os.path.exists(labels_dir):
    sprawdz_spojnosc(images_dir, labels_dir)
else:
    print("BŁĄD: Podane ścieżki do folderów nie istnieją!")