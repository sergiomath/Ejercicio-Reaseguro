

import pandas as pd
import statsmodels.api as sm

# Datos base
data = {
    "Año": [2019, 2020, 2021, 2022, 2023, 2024],
    "Prima": [140599020091, 159266407872, 170076867163,
              167465546189, 177834215618, 194693902188],
    "RSA_Pagos_IBNR": [97801920857, 237645734741, 170763441901,
                       156111077240, 167547514460, 53968670611],
    "LR": [0.70, 1.49, 1.00, 0.93, 0.94, 0.28]
}
df = pd.DataFrame(data)

# === Modelo 1: Prima ===
X = sm.add_constant(df["Año"])
modelo_prima = sm.OLS(df["Prima"], X).fit()
pred_prima_2025 = modelo_prima.get_prediction([[1, 2025]])
print("Prima 2025:")
print(pred_prima_2025.summary_frame(alpha=0.05))  # IC 95%

# === Modelo 2: RSA+Pagos+IBNR ===
for a in [0.05, 0.02, 0.01, 0.025]:
    modelo_rsa = sm.OLS(df["RSA_Pagos_IBNR"], X).fit()
    pred_rsa_2025 = modelo_rsa.get_prediction([[1, 2025]])
    print("\nRSA+Pagos+IBNR 2025:")
    print(f"Con alpha {a}",pred_rsa_2025.summary_frame(alpha=a))

    # === Cálculo de LR proyectado ===
    prima_2025 = pred_prima_2025.predicted_mean[0]
    rsa_2025 = pred_rsa_2025.predicted_mean[0]
    lr_2025 = rsa_2025 / prima_2025
    print(f"\n Con alpha {a} LR 2025 estimado ≈ {lr_2025:.2%}")

    # === Modelo 3 (opcional): LR directo ===
    modelo_lr = sm.OLS(df["LR"], X).fit()
    pred_lr_2025 = modelo_lr.get_prediction([[1, 2025]])
    print("\nLR 2025 (modelo directo):")
    print(f"Con alpha {a}",pred_lr_2025.summary_frame(alpha=a))

prima_ini = df.loc[df["Año"]==2019, "Prima"].values[0]
prima_fin = df.loc[df["Año"]==2024, "Prima"].values[0]
años = 2024 - 2019
cagr = (prima_fin/prima_ini)**(1/años) - 1

# Proyección para 2025
prima_2025 = prima_fin * (1 + cagr)

print(f"CAGR de Prima 2019–2024: {cagr:.2%}")
print(f"Prima proyectada 2025: {prima_2025:,.0f}")