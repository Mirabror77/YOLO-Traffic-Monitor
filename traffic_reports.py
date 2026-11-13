import matplotlib.pyplot as plt
import pandas as pd

# --- SIMULATED DATA ---
data = {
    'Time': ['08:00', '09:00', '10:00', '11:00', '12:00'],
    'Cars': [120, 150, 110, 95, 105],
    'Buses': [15, 20, 10, 5, 8],
    'Trucks': [5, 8, 12, 15, 10],
    'Motorbikes': [45, 50, 30, 25, 40]
}
df = pd.DataFrame(data)

# --- CHART 1: STACKED BAR (Volume) ---
plt.figure(figsize=(10, 6))
plt.bar(df['Time'], df['Cars'], label='Cars', color='#3498db')
plt.bar(df['Time'], df['Buses'], bottom=df['Cars'], label='Buses', color='#2ecc71')
plt.bar(df['Time'], df['Trucks'], bottom=df['Cars']+df['Buses'], label='Trucks', color='#e74c3c')
plt.bar(df['Time'], df['Motorbikes'], bottom=df['Cars']+df['Buses']+df['Trucks'], label='Motorbikes', color='#f39c12')

plt.title('Hourly Traffic Volume (Apex Research)')
plt.legend()
plt.savefig('Report_A.png')
print("Report A Saved.")
plt.show()

# --- CHART 2: PIE CHART (Composition) ---
totals = [df['Cars'].sum(), df['Buses'].sum(), df['Trucks'].sum(), df['Motorbikes'].sum()]
labels = ['Cars', 'Buses', 'Trucks', 'Bikes']
explode = (0, 0, 0.1, 0) # Pop out Trucks

plt.figure(figsize=(8, 8))
plt.pie(totals, labels=labels, autopct='%1.1f%%', explode=explode, startangle=140)
plt.title('Vehicle Composition Split')
plt.savefig('Report_B.png')
print("Report B Saved.")
plt.show()