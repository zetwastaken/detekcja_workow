import pandas as pd
from pathlib import Path

runs_dir = Path('runs/segment')

print("\n" + "="*90)
print("PORÓWNANIE WSZYSTKICH TRENINGÓW YOLOv8")
print("="*90)

results = []
for i in range(1, 6):
    run_folder = runs_dir / f'yolov8_sandbag_seg_v{i}'
    results_file = run_folder / 'results.csv'
    
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            df = df.dropna(subset=['metrics/mAP50-95(M)'])
            
            if not df.empty:
                best_idx = df['metrics/mAP50-95(M)'].idxmax()
                best = df.loc[best_idx]
                
                results.append({
                    'Version': f'v{i}',
                    'Epoch': int(best['epoch']),
                    'mAP50-95': float(best['metrics/mAP50-95(M)']),
                    'mAP50': float(best['metrics/mAP50(M)']),
                    'Precision': float(best['metrics/precision(M)']),
                    'Recall': float(best['metrics/recall(M)']),
                    'Total_Epochs': len(df)
                })
        except Exception as e:
            print(f"❌ Błąd przy v{i}: {e}")

if results:
    results_df = pd.DataFrame(results).sort_values('mAP50-95', ascending=False)
    
    print("\n📊 RANKING TRENINGÓW:\n")
    for idx, row in results_df.iterrows():
        rank = "🥇" if idx == results_df.index[0] else "🥈" if idx == results_df.index[1] else "🥉" if idx == results_df.index[2] else "  "
        print(f"{rank} {row['Version']}: mAP50-95={row['mAP50-95']:.4f} | mAP50={row['mAP50']:.4f} | "
              f"Precision={row['Precision']:.4f} | Recall={row['Recall']:.4f} | "
              f"Best@Epoch {row['Epoch']}/{row['Total_Epochs']}")
    
    print("\n" + "="*90)
    best = results_df.iloc[0]
    print(f"🏆 ZWYCIĘZCA: {best['Version']}")
    print(f"   mAP50-95: {best['mAP50-95']:.4f}")
    print(f"   Najlepsza epoka: {best['Epoch']}")
    print("="*90)
else:
    print("❌ Nie znaleziono wyników treningów!")