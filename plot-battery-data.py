import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_eps_voltage(filepath, test_desc, time_col='ElapsedTime', voltage_col='BatteryVoltage', insun_col='InSun'):

    df = pd.read_csv(filepath)
    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[voltage_col], marker='o', label='Battery Voltage (V)', color='tab:blue', linewidth=2)

    # Shade regions where the satellite is in the dark
    in_dark = df[insun_col] == 0
    starts = np.where((~in_dark.shift(fill_value=False)) & in_dark)[0]
    ends = np.where((in_dark.shift(fill_value=False)) & ~in_dark)[0]
    if in_dark.iloc[-1]:
        ends = np.append(ends, len(df) - 1)
    for start, end in zip(starts, ends):
        plt.axvspan(df[time_col].iloc[start], df[time_col].iloc[end], color='gray', alpha=0.2, label='Eclipse' if start == starts[0] else "")

    plt.title('Battery Voltage vs. Time')
    plt.suptitle(f"Test Scenario: {test_desc}", fontsize=12, y=0.96)
    plt.xlabel(f'{time_col} (seconds)')
    plt.ylabel(f'{voltage_col} (V)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_eps_soc(filepath, test_desc, time_col='ElapsedTime', soc_col='SOC', insun_col='InSun'):

    df = pd.read_csv(filepath)
    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[soc_col], label='SOC', color='tab:orange', linewidth=4)

    # Shade regions where the satellite is in the dark
    in_dark = df[insun_col] == 0
    starts = np.where((~in_dark.shift(fill_value=False)) & in_dark)[0]
    ends = np.where((in_dark.shift(fill_value=False)) & ~in_dark)[0]
    if in_dark.iloc[-1]:
        ends = np.append(ends, len(df) - 1)
    for start, end in zip(starts, ends):
        plt.axvspan(df[time_col].iloc[start], df[time_col].iloc[end], color='gray', alpha=0.2, label='Eclipse' if start == starts[0] else "")

    plt.title('SOC vs. Time')
    plt.suptitle(f"Test Scenario: {test_desc}", fontsize=12, y=0.96)
    plt.xlabel(f'{time_col} (seconds)')
    plt.ylabel(f'{soc_col} (fraction or %)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Special graphing function for ENV 3, as it only logs every 10 seconds
def env3_plot_eps_soc(filepath, test_desc, time_col='ElapsedTime', soc_col='SOC', insun_col='InSun', ymin=None, ymax=None):
    df = pd.read_csv(filepath)
    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[soc_col], color='tab:blue', linewidth=4, label='SOC')

    # Shade regions where the satellite is in the dark
    in_dark = df[insun_col] == 0
    starts = np.where((~in_dark.shift(fill_value=False)) & in_dark)[0]
    ends = np.where((in_dark.shift(fill_value=False)) & ~in_dark)[0]
    if in_dark.iloc[-1]:
        ends = np.append(ends, len(df) - 1)
    for start, end in zip(starts, ends):
        plt.axvspan(df[time_col].iloc[start], df[time_col].iloc[end], color='gray', alpha=0.2, label='Eclipse' if start == starts[0] else "")

    plt.title('SOC vs. Time')
    plt.suptitle(f"Test Scenario: {test_desc}", fontsize=12, y=0.96)
    plt.xlabel(f'{time_col} (seconds)')
    plt.ylabel(f'{soc_col} (fraction or %)')
    plt.grid(True)
    plt.legend()
    # --- FORCE SAME Y-AXIS SCALE ---
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    plot_eps_soc('outputs/nominal_always_sun_charging_log.csv', "Nominal Sun")
    plot_eps_soc('outputs/nominal_night_pass_charging_log.csv', "Nominal Night Pass")
    plot_eps_soc('outputs/test_switch7_log.csv', "Switch 7 On")
    plot_eps_soc('outputs/broken_negY_solar_panel_log_NormalVectors.csv', "Broken -Y Solar Panel - Standard Vectors")
    plot_eps_soc('outputs/solar_degradation_log.csv', "Panel Degradation")
    plot_eps_soc('outputs/rapid_tumble_log.csv', "Rapid Tumbling")

    env3_plot_eps_soc('outputs/ENV3-Nominal.csv', "Environment 3 - Nominal", ymin=0.20, ymax=0.25)
    env3_plot_eps_soc('outputs/ENV3-BrokenNegY.csv', "Environment 3 - Broken -Y Panel", ymin=0.20, ymax=0.25)


    plot_eps_soc('outputs/heavy_negY_biased_broken_solar_panel_log.csv', "Broken Solar Panel - Y Biased Orbit")
