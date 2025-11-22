import jaro
import numpy as np
import pandas as pd
import seaborn as sns
from fuzzywuzzy import fuzz
from matplotlib import colormaps
from matplotlib import pyplot as plt


def pprint_confusion_matrix(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()
    counts = np.array([[TN, FP], [FN, TP]])
    normalized = counts / counts.sum()
    desc = [["TN", "FP"], ["FN", "TP"]]

    def format_html(description, count, norm):
        return (
            f"<span style='font-size:smaller'>{description}</span>"
            f"<br><b>{count}</b><br><span style='font-size:smaller'>{norm:.2f}</span>"
        )

    html_cells = [
        [format_html(desc[i][j], counts[i][j], normalized[i][j]) for j in range(2)]
        for i in range(2)
    ]

    df = pd.DataFrame(
        html_cells,
        index=["GT: ❌", "GT: ✅"],
        columns=["Pred: ❌", "Pred: ✅"],
    )

    flat_norm = normalized.flatten()
    cmap = colormaps["viridis"]
    color_map = cmap(flat_norm)  # RGBA values

    def rgba_to_rgb_str(rgba):
        r, g, b, _ = rgba
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    def get_text_color(rgba):
        r, g, b, _ = rgba
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.5 else "white"

    hex_colors = [rgba_to_rgb_str(c) for c in color_map]
    text_colors = [get_text_color(c) for c in color_map]

    cell_colors = np.array(hex_colors).reshape(2, 2)
    cell_text_colors = np.array(text_colors).reshape(2, 2)

    def style_func(x):
        df_styles = pd.DataFrame("", index=x.index, columns=x.columns)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                bg_color = cell_colors[i, j]
                txt_color = cell_text_colors[i, j]
                df_styles.iloc[i, j] = (
                    f"background-color: {bg_color}; color: {txt_color}; text-align: center;"
                )
        return df_styles

    styled_df = (
        df.style.set_table_attributes(
            "style='border-collapse:collapse; font-family:sans-serif'"
        )
        .set_properties(**{"text-align": "center"})
        .apply(style_func, axis=None)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("font-size", "14px"), ("text-align", "center")],
                }
            ]
        )
    )
    return styled_df


def pprint_metrics(threshold, metrics, metrics_to_print):
    print_metrics = [[threshold] + [metrics[k].item() for k in metrics_to_print]]
    df = pd.DataFrame(
        print_metrics,
        columns=["Threshold"] + [m.replace("_", " ").title() for m in metrics_to_print],
    ).T
    df.columns = [""]
    return df


def plot_metrics(thresholds, metrics, best_threshold, metrics_to_plot=[]):
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))
    for metric_name in metrics_to_plot:
        if metric_name in metrics:
            sns.lineplot(
                x=thresholds,
                y=metrics[metric_name],
                label=metric_name.replace("_", " ").title(),
            )

    plt.axvline(
        best_threshold,
        color="tab:blue",
        linewidth=1,
    )

    plt.xlabel("Distance Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.grid()
    plt.close()

    return fig
